"""One place for every configured value.

A value is decided in this order, earlier wins:

1. Command-line arguments (--project / --output-dir / --dry-run / --guide, and --preset
   and the --no- flags, which are written into the environment)
2. Environment variables (the setting's own name, e.g. DEF_K=20)
3. A .env file in the directory the command is run from, and nowhere else, with the
   same names. It is read once, before any setting is, and fills only the names the
   shell left unset or empty. A .env file git tracks is not read (git_index_state), and
   skip_env_setting_list() names the settings of this tool such a file sets. The
   libraries that look for a .env file of their own when they are imported are imported
   here after it, with that search turned off (_import_without_env_search). A .env file
   is written from the template beside this file (ENV_TEMPLATE_PATH)
4. The PRESET column, for the settings _PRESET_TABLE_DICT lists
5. The defaults in this file

A name set to an empty value is set: env() reads it as the empty string where the setting
gives the empty string a meaning or has it as its default, and as the default otherwise.

The exceptions are PRIMARY_EXT, TEST_COMMAND, TEST_PATH, TEST_PATH_ALT_TUPLE,
TEST_TARGET_STYLE, RUN_COMMAND and CHECK_COMMAND, which set_primary_ext() rebuilds.
"""

import base64
import codecs
import contextlib
import errno
import http.client
import io
import ipaddress
import json
import logging
import math
import os
import posixpath
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import types
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass

import httpx
from dotenv import dotenv_values, load_dotenv
from dotenv import parser as dotenv_parser

from codetwine_loom import ENV_FILE_NAME, SettingError, hold_env_file, is_env_file_hold, \
    tmpfile

logger = logging.getLogger(__name__)

# python-dotenv's own switch, and the values of it that turn loading off. A process
# started with it reads no .env file, here or in a library that loads one when it is
# imported
NO_ENV_FILE_ENV = "PYTHON_DOTENV_DISABLED"
_NO_ENV_FILE_VALUE_TUPLE = ("1", "true", "t", "yes", "y")

# The value NO_ENV_FILE_ENV is set to, to turn loading off
NO_ENV_FILE_VALUE = _NO_ENV_FILE_VALUE_TUPLE[0]


def is_env_file_off() -> bool:
    """Say whether NO_ENV_FILE_ENV turns the loading of .env files off in this process.

    Returns:
        True when the variable holds one of _NO_ENV_FILE_VALUE_TUPLE, whatever its case.
    """
    return os.environ.get(NO_ENV_FILE_ENV, "").casefold() in _NO_ENV_FILE_VALUE_TUPLE


@contextlib.contextmanager
def env_file_off() -> Iterator[None]:
    """Set NO_ENV_FILE_ENV to NO_ENV_FILE_VALUE for the block, then put back what it held.

    Yields:
        None, once. On leaving the block the variable is removed again when it was unset
        before, and set to its earlier value otherwise.

    Raises:
        KeyError: On leaving the block, when the variable was unset before and the block
            removed it.
    """
    switch_before = os.environ.get(NO_ENV_FILE_ENV)
    os.environ[NO_ENV_FILE_ENV] = NO_ENV_FILE_VALUE
    try:
        yield
    finally:
        if switch_before is None:
            del os.environ[NO_ENV_FILE_ENV]
        else:
            os.environ[NO_ENV_FILE_ENV] = switch_before


# The variables that tell git which repository, working tree, index and object store a
# command works on, whatever directory it is run in, and the one that hands
# "git config" alone a file to read and write in place of every file the other git
# commands read their settings from
GIT_PLACE_NAME_TUPLE = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR",
                        "GIT_NAMESPACE", "GIT_CONFIG")

# The variables that change what git answers whatever its command line says: how a
# path it is handed is read, how many lines of context a diff holds, and where the
# attributes of a file are read from
GIT_ANSWER_NAME_TUPLE = ("GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS",
                         "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS", "GIT_DIFF_OPTS",
                         "GIT_ATTR_SOURCE")

# The variables an HTTP client reads its proxies from, each read under this name and
# under the same name in lower case: the ones that hold a proxy's URL, and the one
# that names the hosts reached without a proxy
PROXY_URL_NAME_TUPLE = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")
PROXY_NAME_TUPLE = (*PROXY_URL_NAME_TUPLE, "NO_PROXY")


def _proxy_case_pair_list() -> list[tuple[str, str]]:
    """Pair each proxy variable's name in one case with the same name in the other case.

    Examples:
        PROXY_NAME_TUPLE ("HTTPS_PROXY", ...)
            -> [("HTTPS_PROXY", "https_proxy"), ("https_proxy", "HTTPS_PROXY"), ...]

    Returns:
        For each name of PROXY_NAME_TUPLE, in its order: (the name, the name in lower
        case), then (the name in lower case, the name).
    """
    return [pair for upper_name in PROXY_NAME_TUPLE
            for pair in ((upper_name, upper_name.lower()), (upper_name.lower(), upper_name))]


def _env_fill_skip(name: str, shell_name_set: set[str]) -> bool:
    """Say whether a .env file leaves a name unfilled that the shell holds no value for.

    That is NO_ENV_FILE_ENV, and a proxy variable (PROXY_NAME_TUPLE, in either case)
    the shell holds a value for in the other case.

    Examples:
        ("https_proxy", {"HTTPS_PROXY"})     -> True
        ("https_proxy", set())               -> False
        ("PYTHON_DOTENV_DISABLED", set())    -> True

    Args:
        name: The variable's name.
        shell_name_set: The names the shell holds a value that is not empty for.

    Returns:
        True when the file does not fill name.
    """
    if name in shell_name_set:
        return False
    return name == NO_ENV_FILE_ENV or any(
        other == name and shell_name in shell_name_set
        for shell_name, other in _proxy_case_pair_list())


# The schemes of the proxies the HTTP clients of this tool go through
PROXY_SCHEME_TUPLE = ("http", "https")

# The keys of urllib.request.getproxies() the HTTP clients of this tool take a proxy
# from, in the order they are looked at
_PROXY_KEY_TUPLE = ("https", "http", "all")

# The host a proxy's URL is read with in place of its own, to tell which part of the
# URL an HTTP client cannot read
_STAND_IN_PROXY_HOST = "proxy.invalid"

# The endpoint that stands for the providers' own ones in the check of NO_PROXY, a host
# no entry names, and the name it is said under
_PROVIDER_ENDPOINT_URL = "https://ctloom-endpoint.invalid/"
_PROVIDER_ENDPOINT_NAME = "the providers' own endpoints"

# The models _model_endpoint() has said it did not ask litellm the endpoint of. Another
# import of this module in the process says it of none of them again
_UNASKED_ENDPOINT_MODEL_SET: set[str] = globals().get("_UNASKED_ENDPOINT_MODEL_SET", set())

# The names of the variables counted as the providers' endpoints: those that end in
# _API_BASE, _URL or _ENDPOINT. Their values are counted when they start with these
# schemes, in any case
_ENDPOINT_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*_(?:API_BASE|URL|ENDPOINT)")
_ENDPOINT_SCHEME_TUPLE = ("http://", "https://")

# What a NO_PROXY entry the HTTP clients read differently is, by its kind, and how to
# write it instead
_NO_PROXY_ENTRY_TEXT_DICT = {
    "scheme": ("an entry with a scheme (scheme://host)", "write the host alone"),
    "port": ("an entry with a port (host:port)", "write the host alone"),
    "dot": ("an entry with a leading '.' that names the host itself",
            "write the host without the '.'"),
    "star": ("'*' among other entries", "write '*' alone, or leave it out"),
    "other": ("an entry the clients match differently",
              "write the host as the URL writes it"),
}

# The variable an HTTP client reads the file of the certificate authorities it trusts
# from, and the one the requests library and botocore read it from
CA_FILE_ENV = "SSL_CERT_FILE"
REQUESTS_CA_FILE_ENV = "REQUESTS_CA_BUNDLE"

# How long git is given to answer git_index_state(), in seconds
_GIT_ANSWER_TIMEOUT_SEC = 5

# How many files git looks through for one that moved, in a diff and in a merge
GIT_RENAME_LIMIT = 7000

# The settings a git command is given on its command line ("-c"), whatever the user's
# and the repository's own configuration say: no automatic gc, no hook, no file system
# monitor program, a path with non-ASCII characters written as itself, a file whose time
# or size differs from the index compared by its contents, no entry of the index marked
# as one whose file is not looked at, a commit's message written out in UTF-8, a trailer
# read as a name, ":" and a value, no line about a commit's signature in what a log
# writes, a file read as binary or as text by its contents where no attribute says
# which, and a file that moved looked for among as many as GIT_RENAME_LIMIT files
GIT_SETTING_TUPLE = ("-c", "gc.auto=0", "-c", f"core.hooksPath={os.devnull}",
                     "-c", "core.quotePath=false", "-c", "core.fsmonitor=false",
                     "-c", "diff.autoRefreshIndex=true", "-c", "core.ignoreStat=false",
                     "-c", "i18n.logOutputEncoding=UTF-8", "-c", "trailer.separators=:",
                     "-c", "log.showSignature=false", "-c", "diff.default.binary=auto",
                     "-c", f"diff.renameLimit={GIT_RENAME_LIMIT}")

# What git_index_state() answers: the index names the file, it does not, and git did
# not say
INDEX_HOLD = "hold"
INDEX_FREE = "free"
INDEX_UNKNOWN = "unknown"

# What stands in ENV_FILE_SKIP_STATE for a .env file NO_ENV_FILE_ENV keeps from being read,
# and for one hold_env_file() keeps from being read
ENV_FILE_OFF = "off"
ENV_FILE_HOLD = "setup"


def _in_git_tree(directory: str) -> bool:
    """Say whether a directory, or one above it, holds an entry named .git.

    Args:
        directory: The directory, as an absolute path. It does not have to be there.

    Returns:
        True when one does.
    """
    while True:
        if os.path.lexists(os.path.join(directory, ".git")):
            return True
        parent = os.path.dirname(directory)
        if parent == directory:
            return False
        directory = parent


def git_index_state(path: str) -> str:
    """Say whether git tracks a file at path, whatever the case of the letters of its name.

    git is asked in the directory of path, with GIT_SETTING_TUPLE and without
    GIT_PLACE_NAME_TUPLE and GIT_ANSWER_NAME_TUPLE in its environment: the repository is
    the one that directory is in, and the name is read as it is written. The name is
    compared without regard to case: a file the index names ".ENV" is one it names for
    ".env".

    Args:
        path: The file, as an absolute path. Neither it nor its directory has to be
            there.

    Returns:
        INDEX_HOLD when the index of the repository path is in names it. INDEX_FREE
        when it does not, when git is not installed, and when git gives no answer in a
        directory with no .git in it or above it. INDEX_UNKNOWN when git gives no
        answer (it cannot be started in the directory, fails, or does not end within
        _GIT_ANSWER_TIMEOUT_SEC) in a directory with a .git in it or above it.
    """
    if shutil.which("git") is None:
        return INDEX_FREE
    directory = os.path.dirname(path)
    env_dict = {name: value for name, value in os.environ.items()
                if name not in GIT_PLACE_NAME_TUPLE + GIT_ANSWER_NAME_TUPLE}
    try:
        proc = subprocess.run(
            ["git", *GIT_SETTING_TUPLE, "ls-files", "-z", "--",
             f":(icase,literal){os.path.basename(path)}"],
            cwd=directory, env=env_dict, stdin=subprocess.DEVNULL,
            capture_output=True, check=False, timeout=_GIT_ANSWER_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0:
        return INDEX_HOLD if proc.stdout else INDEX_FREE
    return INDEX_UNKNOWN if _in_git_tree(directory) else INDEX_FREE


# What git_index_state() answered for each .env file this process asked it about, by
# the file's path. Another import of this module in the process asks for none of them
# again
_ENV_FILE_STATE_DICT: dict[str, str] = globals().get("_ENV_FILE_STATE_DICT", {})


def accept_env_file(path: str) -> None:
    """Have the next import of this module read the .env file at path, git not asked.

    Args:
        path: A .env file this process wrote, as an absolute path.
    """
    _ENV_FILE_STATE_DICT[path] = INDEX_FREE


def _env_file_path() -> tuple[str, str, str]:
    """Decide which .env file this process reads.

    The .env file is the one in the directory the command is run from, and none other:
    not one in a directory above it, and not one where this package is installed. One
    git tracks, and one git does not say whether it tracks, is left unread, and so is
    any while NO_ENV_FILE_ENV turns loading off and one hold_env_file() was called for,
    git not asked. git is asked about a file once in a process (_ENV_FILE_STATE_DICT).

    Returns:
        (the file to read, the file left unread, what left it unread:
        git_index_state()'s answer, ENV_FILE_OFF or ENV_FILE_HOLD), the files each as
        an absolute path or "". All are "" when there is no .env file.
    """
    if not os.path.isfile(ENV_FILE_NAME):
        return "", "", ""
    path = os.path.abspath(ENV_FILE_NAME)
    if is_env_file_hold(path):
        return "", path, ENV_FILE_HOLD
    if is_env_file_off():
        return "", path, ENV_FILE_OFF
    if path not in _ENV_FILE_STATE_DICT:
        _ENV_FILE_STATE_DICT[path] = git_index_state(path)
    state = _ENV_FILE_STATE_DICT[path]
    return (path, "", "") if state == INDEX_FREE else ("", path, state)


# Each name a .env file this process read filled, mapped to (the value the shell held
# for it, None when it held none; the value the file gave it). Another import of this
# module in the process keeps them
_ENV_FILE_FILL_DICT: dict[str, tuple[str | None, str]] = globals().get(
    "_ENV_FILE_FILL_DICT", {})


def _load_env_file(path: str) -> None:
    """Fill the names the shell left unset or empty with what the .env file at path holds.

    A name the shell holds a value for is left as it is. A name the shell holds empty
    and the file does not name stays empty. NO_ENV_FILE_ENV is not filled, and neither
    is a proxy variable (PROXY_NAME_TUPLE) whose name in the other case, upper or lower,
    the shell holds a value for (_env_fill_skip). Each name filled is kept in
    _ENV_FILE_FILL_DICT.

    Args:
        path: The .env file, as an absolute path.

    Raises:
        SettingError: Naming the file, when it cannot be read, is not UTF-8, or holds
            a name or a value the environment does not take (a NUL character).
    """
    shell_dict = dict(os.environ)
    empty_name_list = [name for name, value in os.environ.items() if value == ""]
    for name in empty_name_list:
        del os.environ[name]
    shell_name_set = set(os.environ)
    try:
        load_dotenv(path)
        for name in (NO_ENV_FILE_ENV, *(other for _, other in _proxy_case_pair_list())):
            if _env_fill_skip(name, shell_name_set):
                os.environ.pop(name, None)
        _ENV_FILE_FILL_DICT.update({name: (shell_dict.get(name), value)
                                    for name, value in os.environ.items()
                                    if shell_dict.get(name) != value})
    except (OSError, ValueError) as e:
        raise SettingError(
            f"{path} could not be read as the .env file ({type(e).__name__}: {e}). "
            f"Make it a UTF-8 file with no NUL character in it that this user can "
            f"read, or remove it") from None
    finally:
        for name in empty_name_list:
            os.environ.setdefault(name, "")


# Each name the command line wrote into this process's environment, mapped to (the
# value the environment held for it before, None when it held none; the value the
# command line gave it). Another import of this module in the process keeps them
_OPTION_FILL_DICT: dict[str, tuple[str | None, str]] = globals().get(
    "_OPTION_FILL_DICT", {})


def keep_option_value(name: str, prior_value: str | None, value: str) -> None:
    """Keep in _OPTION_FILL_DICT that the command line gives name value over prior_value.

    prior_value is what the environment holds for name before the command line sets it.

    Args:
        name: The setting's name.
        prior_value: The environment's value for name; None when it holds none.
        value: The value the command line gives.
    """
    _OPTION_FILL_DICT[name] = (prior_value, value)


# Each name this module wrote into or took out of this process's environment for
# itself and for the libraries it imports, mapped to (the value the environment held
# for it before, None when it held none; the value written, None when the name was
# taken out). Another import of this module in the process keeps them
_OWN_FILL_DICT: dict[str, tuple[str | None, str | None]] = globals().get(
    "_OWN_FILL_DICT", {})


def _keep_own_value(name: str, prior_value: str | None, value: str | None) -> None:
    """Keep in _OWN_FILL_DICT that this module changes name from prior_value to value.

    Nothing is kept when the two are the same. A name changed before, whose value is
    still the one written then, keeps the value it held before that change, and is
    left out once it is written back to that value.

    Args:
        name: The variable's name.
        prior_value: The environment's value for name; None when it holds none.
        value: The value written; None when the name is taken out.
    """
    if prior_value == value:
        return
    earlier = _OWN_FILL_DICT.get(name)
    if earlier is not None and earlier[1] == prior_value:
        prior_value = earlier[0]
    if prior_value == value:
        del _OWN_FILL_DICT[name]
    else:
        _OWN_FILL_DICT[name] = (prior_value, value)


def own_name_set() -> set[str]:
    """Return the names this module wrote into or took out of this process's environment.

    They are the names of _OWN_FILL_DICT: those changed for this module itself and for
    the libraries it imports.

    Returns:
        The names.
    """
    return set(_OWN_FILL_DICT)


def run_value(name: str, answer_dict: Mapping[str, str]) -> str:
    """Return the value a run reads for name once answer_dict is written to .env.

    The environment's value wins when it is not empty (_load_env_file() fills only the
    names the shell left unset or empty); otherwise the answer, unless the file would
    leave the name unfilled (_env_fill_skip).

    Examples:
        LLM_API_BASE=http://127.0.0.1:9 in the shell,
        {"LLM_API_BASE": "http://llm.corp.example/v1"} -> "http://127.0.0.1:9"
        LLM_API_BASE unset or empty -> "http://llm.corp.example/v1"
        HTTPS_PROXY=http://proxy.corp.example:3128 in the shell,
        {"https_proxy": "http://old.corp.example:3128"}, "https_proxy" -> ""

    Args:
        name: The variable's name.
        answer_dict: The answers the file would hold, by name.

    Returns:
        The value; "" when neither holds one.
    """
    shell_name_set = {shell_name for shell_name, value in os.environ.items() if value}
    if name in shell_name_set:
        return os.environ[name]
    return "" if _env_fill_skip(name, shell_name_set) else answer_dict.get(name, "")


def proxy_run_value(name: str, answer_dict: Mapping[str, str]) -> str:
    """Return the value the HTTP clients of a run take for a proxy variable.

    The variable is read under its name in lower case and in upper case (run_value),
    and the name in lower case is taken first, as urllib.request.getproxies() takes it.

    Examples:
        {"HTTPS_PROXY": "http://new:3128", "https_proxy": "http://old:3128"}
            -> "http://old:3128"
        {"HTTPS_PROXY": "http://new:3128"} -> "http://new:3128"

    Args:
        name: A name of PROXY_NAME_TUPLE.
        answer_dict: The answers the file would hold, by name.

    Returns:
        The value; "" when neither name has one.
    """
    return run_value(name.lower(), answer_dict) or run_value(name.upper(), answer_dict)


def run_setting_value(name: str, answer_dict: Mapping[str, str]) -> str:
    """Return the value a run takes for a variable once answer_dict is written to .env.

    A proxy variable (PROXY_NAME_TUPLE) is the value the HTTP clients take
    (proxy_run_value). Any other is run_value() as its setting reads it
    (setting_form).

    Examples:
        {"WEB_SEARCH_PROVIDER": " Tavily"} -> "tavily"
        {"NO_PROXY": "a.example", "no_proxy": "b.example"}, "NO_PROXY" -> "b.example"

    Args:
        name: The variable's name.
        answer_dict: The answers the file would hold, by name.

    Returns:
        The value; "" when neither the environment nor answer_dict holds one.
    """
    if name in PROXY_NAME_TUPLE:
        return proxy_run_value(name, answer_dict)
    return setting_form(name, run_value(name, answer_dict))


def setting_form(name: str, value: str) -> str:
    """Return a value as its setting reads it.

    A setting that takes one of a few values (choice_tuple_of) reads the one value
    names, its case and the blanks around it aside (choice_of). Any other setting reads
    value as it is.

    Examples:
        ("PRESET", " High") -> "high"
        ("PRESET", "hgih")  -> "hgih"
        ("LLM_API_BASE", " http://llm/v1") -> " http://llm/v1"

    Args:
        name: The setting's name.
        value: The value as it is written.

    Returns:
        The member of the setting's values value names, or value itself when the
        setting takes any value or value names none of them.
    """
    choice_tuple = choice_tuple_of(name)
    if not choice_tuple:
        return value
    try:
        return choice_of(name, value, choice_tuple)
    except SettingError:
        return value


# ENV_FILE_PATH is the .env file read, ENV_FILE_SKIP_PATH the one left unread, and
# ENV_FILE_SKIP_STATE what left it unread: what git_index_state() answers for it,
# ENV_FILE_OFF or ENV_FILE_HOLD. What the file read holds fills in the names the shell
# left unset or empty
ENV_FILE_PATH, ENV_FILE_SKIP_PATH, ENV_FILE_SKIP_STATE = _env_file_path()


def env_file_skip_text() -> str:
    """Say in a few words what kept ENV_FILE_SKIP_PATH from being read.

    Returns:
        The words, to follow the path of the file. "" when no file was left unread.
    """
    if not ENV_FILE_SKIP_PATH:
        return ""
    if ENV_FILE_SKIP_STATE == ENV_FILE_OFF:
        return f"was not read: {NO_ENV_FILE_ENV} turns the loading of .env files off"
    if ENV_FILE_SKIP_STATE == ENV_FILE_HOLD:
        return "is read and written by setup alone"
    if ENV_FILE_SKIP_STATE == INDEX_HOLD:
        return "is tracked by git and was not read"
    return "is in a git repository git did not answer for and was not read"


if ENV_FILE_PATH:
    _load_env_file(ENV_FILE_PATH)


def _drop_empty_proxy_name() -> None:
    """Unset a proxy variable that is empty in one case while the other case holds a value.

    Each name unset is kept in _OWN_FILL_DICT.

    Examples:
        HTTPS_PROXY=http://proxy.corp.example:3128, https_proxy= (empty)
            -> https_proxy is unset
        https_proxy= (empty), HTTPS_PROXY unset -> left as they are
    """
    for name, other in _proxy_case_pair_list():
        if os.environ.get(name) == "" and os.environ.get(other):
            _keep_own_value(name, "", None)
            del os.environ[name]


def ca_file_error(path: str) -> str:
    """Say what keeps a file from being read as certificate authorities.

    Args:
        path: The file.

    Returns:
        "" when the file holds at least one PEM certificate an HTTP client loads.
        Otherwise what the loading failed with, in a few words.
    """
    try:
        ssl.create_default_context(cafile=path)
    except (OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return ""


def ca_file_path_error(path: str) -> str:
    """Say what keeps a certificate file from being mounted in a container.

    Args:
        path: The file, as an absolute path.

    Returns:
        "" when the path can be named in the engine's mount argument; otherwise why
        not, in a few words.
    """
    if ":" in path:
        return "a path holding a ':' cannot be given to the container that fetches dependencies"
    return ""


def _settle_ca_file() -> None:
    """Check the file CA_FILE_ENV names and hand it on under REQUESTS_CA_FILE_ENV too.

    Nothing is done when CA_FILE_ENV is unset or empty. Otherwise CA_FILE_ENV is set to
    the file's absolute path, and REQUESTS_CA_FILE_ENV to the same path when it holds no
    value. Each name changed is kept in _OWN_FILL_DICT.

    Raises:
        SettingError: When CA_FILE_ENV names no file, or a file that holds no PEM
            certificate (ca_file_error).
    """
    ca_file = os.environ.get(CA_FILE_ENV, "")
    if not ca_file:
        return
    if not os.path.isfile(ca_file):
        raise SettingError(
            f"{CA_FILE_ENV} names {ca_file}, which is not a file. Write the path of "
            f"a PEM file out in full (a leading ~ is not expanded), or unset it")
    ca_file = os.path.abspath(ca_file)
    error = ca_file_error(ca_file)
    if error:
        raise SettingError(
            f"{CA_FILE_ENV} names {ca_file}, which cannot be read as certificate "
            f"authorities ({error}). Point it at a PEM file (-----BEGIN "
            f"CERTIFICATE-----), or unset it")
    _keep_own_value(CA_FILE_ENV, os.environ.get(CA_FILE_ENV), ca_file)
    os.environ[CA_FILE_ENV] = ca_file
    if not os.environ.get(REQUESTS_CA_FILE_ENV):
        _keep_own_value(REQUESTS_CA_FILE_ENV, os.environ.get(REQUESTS_CA_FILE_ENV),
                        ca_file)
        os.environ[REQUESTS_CA_FILE_ENV] = ca_file


_drop_empty_proxy_name()
_settle_ca_file()


def _import_without_env_search() -> None:
    """Import the libraries that load a .env file when imported, with NO_ENV_FILE_ENV set.

    The file such a library loads is one it looks for from the directory it is installed
    in upwards. Imported here, the libraries read the environment as the .env file above
    left it and load no file of their own. NO_ENV_FILE_ENV is put back to what it held
    once they are imported. Each name the imports change is kept in _OWN_FILL_DICT, as
    it was before the imports and as it is after them.
    """
    before_dict = dict(os.environ)
    switch_before = os.environ.get(NO_ENV_FILE_ENV)
    os.environ[NO_ENV_FILE_ENV] = NO_ENV_FILE_VALUE
    try:
        import codetwine.config.settings  # noqa: F401
        import litellm  # noqa: F401
    finally:
        if switch_before is None:
            del os.environ[NO_ENV_FILE_ENV]
        else:
            os.environ[NO_ENV_FILE_ENV] = switch_before
        for name in before_dict.keys() | os.environ.keys():
            _keep_own_value(name, before_dict.get(name), os.environ.get(name))


_import_without_env_search()

# Imported below the .env file: each imports codetwine, which reads its settings from
# the environment when it is imported
from codetwine.config.settings import DOC_EFFORT_TUPLE
from codetwine.utils import file_utils as codetwine_file_utils

from codetwine_loom.languages import LANG_DICT

# The template a .env is written from: the least that has to be set and the settings
# that are changed most often, each with a comment above it
ENV_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), ".env.example")

# A setting line of the template: NAME=value, or the same commented out
_TEMPLATE_LINE_RE = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]*)=(.*)$")

# A question line of the template, above the setting it asks for: "#? y=<value> <question>".
# The setting is asked for with the question and [y/N], and y sets it to <value>
_TEMPLATE_QUESTION_RE = re.compile(r"^#\?\s*y=(\S+)\s+(.+)$")

# A line of the template above a setting that is not asked for: "#? skip". The setting
# keeps the template's line, and so its default, until it is set by hand
_TEMPLATE_SKIP_RE = re.compile(r"^#\?\s*skip\s*$")

# The name of each setting read so far
_ENV_NAME_SET: set[str] = set()

# The default written in this file for each setting env() reads, by its name
_DEFAULT_DICT: dict[str, str | int | float | bool] = {}

# The lowest value each numeric setting env() reads takes, by its name
_LOW_DICT: dict[str, int | float] = {}

# The settings that take one of a few values, with those values, added as each is read
_CHOICE_DICT: dict[str, tuple[str, ...]] = {}


def choice_tuple_of(name: str) -> tuple[str, ...]:
    """Return the values a setting read with choice_env() takes.

    Args:
        name: The setting's name.

    Returns:
        The values, or () for a setting not read with choice_env().
    """
    return _CHOICE_DICT.get(name, ())


def env_name_tuple() -> tuple[str, ...]:
    """Return the names of the settings this module has read.

    The set is filled while this module is imported and is complete once the import has
    finished.

    Returns:
        The names, sorted.
    """
    return tuple(sorted(_ENV_NAME_SET))


def setting_value_dict(suffix: str) -> dict[str, object]:
    """Return the current value of every setting whose name ends in suffix.

    Examples:
        setting_value_dict("_MODEL") -> {"ANSWER_MODEL": "anthropic/...", ...}

    Args:
        suffix: The end of the setting names, e.g. "_MODEL" or "_MAX_TOKEN".

    Returns:
        {setting name: its value in this module}, for the names env_name_tuple() holds.
    """
    module_dict = globals()
    return {name: module_dict[name] for name in env_name_tuple()
            if name.endswith(suffix) and name in module_dict}


@dataclass(frozen=True)
class EnvTemplateEntry:
    """One setting the template names.

    Attributes:
        name: The setting's name.
        value: The value as the template writes it, commented out or not.
        help_list: The comment lines directly above it, without the leading "# ".
        question: The question it is asked for with, or "" when it is asked for by name.
        yes_value: What a y to question sets it to. "" when there is no question.
        skip: True for a setting that is not asked for (a "#? skip" line above it).
    """
    name: str
    value: str
    help_list: list[str]
    question: str = ""
    yes_value: str = ""
    skip: bool = False


# The logger python-dotenv names the lines of a file it cannot read on
_DOTENV_LOGGER_NAME = "dotenv.main"


def skip_env_setting_list() -> list[str]:
    """Return the settings of this tool the .env file git kept from being read would set.

    The file is read for its names alone; no value in it is expanded or put in the
    environment, and python-dotenv writes nothing about the lines it cannot read.

    Returns:
        The names ENV_FILE_SKIP_PATH sets that are a setting this module reads
        (env_name_tuple()), a setting the template names other than the proxy variables
        (PROXY_NAME_TUPLE) and CA_FILE_ENV, or one of the providers' own variables the
        models of the settings read (llm.provider_variable_tuple()), without the names
        the shell holds a value for. Sorted. Empty when no file was left unread, when
        NO_ENV_FILE_ENV or hold_env_file() is what left it unread, and when the file
        cannot be opened.
    """
    if not ENV_FILE_SKIP_PATH or ENV_FILE_SKIP_STATE in (ENV_FILE_OFF, ENV_FILE_HOLD):
        return []
    from codetwine_loom import llm
    dotenv_logger = logging.getLogger(_DOTENV_LOGGER_NAME)
    level = dotenv_logger.level
    dotenv_logger.setLevel(logging.ERROR)
    try:
        with open(ENV_FILE_SKIP_PATH, encoding="utf-8", errors="replace") as handle:
            name_set = set(dotenv_values(stream=handle, interpolate=False))
    except OSError as e:
        logger.warning("%s, which is not read as the .env file, could not be opened to "
                       "see which settings it holds: %s", ENV_FILE_SKIP_PATH, e)
        return []
    finally:
        dotenv_logger.setLevel(level)
    template_name_set = {entry.name for entry in env_template_entry_list()} \
        - {*PROXY_NAME_TUPLE, CA_FILE_ENV}
    setting_name_set = set(env_name_tuple()) | set(llm.provider_variable_tuple()) \
        | template_name_set
    return sorted(name for name in name_set & setting_name_set if not os.environ.get(name))


def _template_line_list() -> list[str]:
    """Return the lines of the template (ENV_TEMPLATE_PATH), read as UTF-8."""
    with open(ENV_TEMPLATE_PATH, encoding="utf-8") as handle:
        return handle.read().splitlines()


def env_template_entry_list() -> list[EnvTemplateEntry]:
    """Return the settings the template names, in the order they are written.

    A blank line ends a block of comment lines; the file's heading is attached to no
    setting. A question line ("#? y=<value> <question>") and a skip line ("#? skip") go
    to the setting below it and not into its help lines.

    Returns:
        One EnvTemplateEntry per setting.
    """
    entry_list, help_list = [], []
    question, yes_value, skip = "", "", False
    for line in _template_line_list():
        match = _TEMPLATE_LINE_RE.match(line)
        question_match = _TEMPLATE_QUESTION_RE.match(line)
        if match:
            entry_list.append(EnvTemplateEntry(match[2], match[3], help_list,
                                               question, yes_value, skip))
            help_list, question, yes_value, skip = [], "", "", False
        elif question_match:
            yes_value, question = question_match[1], question_match[2]
        elif _TEMPLATE_SKIP_RE.match(line):
            skip = True
        elif line.startswith("#"):
            help_list.append(line.lstrip("# "))
        else:
            help_list, question, yes_value, skip = [], "", "", False
    return entry_list


# The values a .env line holds as they are. Any other value is written in quotes
_ENV_BARE_VALUE_RE = re.compile(r"[\w./:@%+,=~-]+")


def _env_value_text(value: str) -> str:
    """Return a value in the form a .env line holds it in, which reads back as value.

    Examples:
        "sk-1"            -> "sk-1"
        "/home/u/my #1"   -> "'/home/u/my #1'"
        "it's"            -> "'it\\'s'"
        "http://h/${X}/"  -> "'http://h/${:-$}{X}/'"

    Args:
        value: The value of one setting.

    Returns:
        value itself when it holds nothing but letters, digits and "_./:@%+,=~-".
        Otherwise value in single quotes, with each backslash and single quote in it
        behind a backslash and each "${" written as "${:-$}{".
    """
    if _ENV_BARE_VALUE_RE.fullmatch(value):
        return value
    quote_text = value.replace("\\", "\\\\").replace("'", "\\'")
    return "'" + quote_text.replace("${", "${:-$}{") + "'"


def _write_private_file(path: str, content: bytes) -> None:
    """Write content to path through tmpfile.atomic_write(), readable by its owner alone.

    Args:
        path: The file to write.
        content: What the file holds.

    Raises:
        OSError: When the file cannot be written.
    """
    with tmpfile.atomic_write(path, 0o600) as tmp_path:
        with open(tmp_path, "wb") as handle:
            handle.write(content)


def write_env_file(path: str, answer_dict: dict[str, str]) -> None:
    """Write the template to path, with the settings in answer_dict set to their values.

    A setting line whose name is in answer_dict becomes NAME=value, no longer commented
    out; every other line is written as the template has it. A name in answer_dict the
    template has no line for is written as NAME=value after the template's last line.
    Each value is written as _env_value_text() gives it. The file is readable by its
    owner alone. The whole text is encoded before anything is made, and written through
    tmpfile.atomic_write(): under a temporary name made readable by its owner alone,
    then moved to path. Whatever stops the write leaves nothing at path.

    Args:
        path: Where to write. A file already there is not written over.
        answer_dict: The values to set, by setting name.

    Raises:
        FileExistsError: When path is already there.
        ValueError: When a value holds a character UTF-8 cannot write (a half of a
            surrogate pair).
        OSError: When the file cannot be written.
    """
    line_list = []
    template_name_set = set()
    for line in _template_line_list():
        match = _TEMPLATE_LINE_RE.match(line)
        if match:
            template_name_set.add(match[2])
        if match and match[2] in answer_dict:
            line = f"{match[2]}={_env_value_text(answer_dict[match[2]])}"
        line_list.append(line)
    line_list += [f"{name}={_env_value_text(value)}"
                  for name, value in answer_dict.items()
                  if name not in template_name_set]
    content = ("\n".join(line_list) + "\n").encode("utf-8")
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, "already there", path)
    _write_private_file(path, content)


# The blank lines python-dotenv puts at the start of the text of the setting below them:
# the blanks it reads there (\s), up to the last line ending among them
_BLANK_LEAD_RE = re.compile(r"\s*(?:\r\n|\n|\r)")

# The line ending at the end of a text
_LINE_END_RE = re.compile(r"(?:\r\n|\n|\r)\Z")

# The byte order mark a UTF-8 file may start with
_BOM_TEXT = "\ufeff"


def env_file_value_dict(path: str) -> dict[str, str]:
    """Return the settings the .env file at path sets to a value that is not empty.

    The values are read as load_dotenv() reads them.

    Args:
        path: The .env file, as an absolute path.

    Returns:
        {name: value}, in the order the file writes them.

    Raises:
        SettingError: Naming the file, when it cannot be read or is not UTF-8.
    """
    try:
        value_dict = dotenv_values(path, encoding="utf-8")
    except (OSError, ValueError) as e:
        raise SettingError(
            f"{path} could not be read as the .env file ({type(e).__name__}: {e}). "
            f"Make it a UTF-8 file this user can read, or remove it") from None
    return {name: value for name, value in value_dict.items() if value}


def _template_line_dict() -> dict[str, str]:
    """Return the template's line for each setting it names, by the setting's name."""
    line_dict = {}
    for line in _template_line_list():
        match = _TEMPLATE_LINE_RE.match(line)
        if match:
            line_dict.setdefault(match[2], line)
    return line_dict


def _line_end(text: str) -> str:
    """Return the line ending text ends with: "\\r\\n", "\\n" or "\\r"; "" for none."""
    match = _LINE_END_RE.search(text)
    return match[0] if match else ""


def _main_line_end(text: str) -> str:
    """Return the line ending the text of a .env file uses most.

    Examples:
        "A=1\\r\\nB=2\\r\\nC=3\\n" -> "\\r\\n"
        "A=1"                 -> "\\n"

    Args:
        text: The text of the file.

    Returns:
        "\\n", "\\r\\n" or "\\r", the first of them in that order when two are used as
        often; "\\n" when text has none.
    """
    crlf_count = text.count("\r\n")
    count_dict = {"\n": text.count("\n") - crlf_count, "\r\n": crlf_count,
                  "\r": text.count("\r") - crlf_count}
    return max(count_dict, key=count_dict.__getitem__)


def _env_binding_list(text: str) -> list[tuple[dotenv_parser.Binding, str]]:
    """Return what python-dotenv reads in the text of a .env file, with the text of each.

    A "\\r\\n" python-dotenv splits between a line it reads no setting in and the next
    line is put whole at the end of the first.

    Args:
        text: The text of the file, its line endings as the file has them.

    Returns:
        (binding, its text) per binding; the texts, joined, are text without the byte
        order mark it starts with.
    """
    binding_list = list(dotenv_parser.parse_stream(io.StringIO(text)))
    piece_list = [binding.original.string for binding in binding_list]
    for index in range(len(piece_list) - 1):
        if piece_list[index].endswith("\r") and piece_list[index + 1].startswith("\n"):
            piece_list[index] += "\n"
            piece_list[index + 1] = piece_list[index + 1][1:]
    return list(zip(binding_list, piece_list))


def rewrite_env_file(path: str, current_dict: Mapping[str, str],
                     answer_dict: Mapping[str, str]) -> bool:
    """Write over the .env file at path the settings answer_dict changes from current_dict.

    A value changes when its setting reads it otherwise (setting_form): " High" and
    "high" are the same PRESET. Every other line is left as it is, its line ending
    ("\\r\\n", "\\n" or "\\r") and the byte order mark the file starts with among it. A
    line written in place of another takes that line's line ending, and a line added
    the one the file uses most (_main_line_end); a last line with no line ending takes
    one only when a line is added after it. A name whose value changes takes NAME=value
    in place of the first line that sets it, and the later lines that set it are taken
    out. A name no line sets is written at the first line that names it commented out
    (# NAME=...): in that line's place when it is the template's line for the name or
    names no value, and after it otherwise, the line kept. With no such line it is
    written after the last line. A name current_dict holds and answer_dict does not
    takes the template's line for it in place of the first line that sets it, or loses
    its lines when the template does not name it or a line names it commented out. Each value is written as _env_value_text() gives it. The file is
    written through tmpfile.atomic_write(), readable by its owner alone.

    Examples:
        "LLM_API_KEY=sk-1", {"LLM_API_KEY": "sk-1"} -> {"LLM_API_KEY": "sk-2"}
            -> "LLM_API_KEY=sk-2"
        "# HTTPS_PROXY=", {} -> {"HTTPS_PROXY": "http://proxy:3128"}
            -> "HTTPS_PROXY=http://proxy:3128"
        "# PRESET=low", {} -> {"PRESET": "high"}
            -> "# PRESET=low\nPRESET=high"
        "PRESET=high", {"PRESET": "high"} -> {}
            -> "# PRESET=medium"

    Args:
        path: The .env file, as an absolute path.
        current_dict: The values the file holds, as env_file_value_dict() gives them.
        answer_dict: The values the file is to hold, by name.

    Returns:
        True when the file was written; False when answer_dict changes nothing.

    Raises:
        ValueError: When a value holds a character UTF-8 cannot write (a half of a
            surrogate pair).
        OSError: When the file cannot be read or written.
    """
    set_dict = {name: value for name, value in answer_dict.items()
                if name not in current_dict
                or setting_form(name, current_dict[name]) != setting_form(name, value)}
    drop_set = {name for name in current_dict if name not in answer_dict}
    if not set_dict and not drop_set:
        return False
    template_line_dict = _template_line_dict()
    with open(path, encoding="utf-8", newline="") as handle:
        file_text = handle.read()
    newline = _main_line_end(file_text)
    pair_list = _env_binding_list(file_text)
    key_set = {binding.key for binding, _ in pair_list if binding.key}
    comment_name_set = {match[2] for match in (
        _TEMPLATE_LINE_RE.match(piece.strip())
        for binding, piece in pair_list if not binding.key) if match and match[1]}
    text_list, write_set = [], set()
    for binding, piece in pair_list:
        lead_match = _BLANK_LEAD_RE.match(piece)
        lead = lead_match[0] if lead_match else ""
        line = piece[len(lead):]
        line_end = _line_end(line)
        name = binding.key
        keep_line = ""
        # A line that names a setting commented out takes the setting no line sets: in
        # its place when it is the template's line or names no value, after it otherwise
        if name is None:
            match = _TEMPLATE_LINE_RE.match(line.strip())
            if match and match[1] and match[2] in set_dict \
                    and match[2] not in key_set | write_set:
                name = match[2]
                template_match = _TEMPLATE_LINE_RE.match(template_line_dict.get(name, ""))
                template_value = template_match[3] if template_match else ""
                if match[3].rstrip() not in ("", template_value):
                    keep_line = line if line_end else line + newline
                    line_end = newline
        if name in set_dict:
            line = "" if name in write_set \
                else f"{keep_line}{name}={_env_value_text(set_dict[name])}{line_end}"
            write_set.add(name)
        elif name in drop_set and binding.key:
            line = "" if name in write_set | comment_name_set \
                or name not in template_line_dict else template_line_dict[name] + line_end
            write_set.add(name)
        text_list.append(lead + line)
    text = "".join(text_list)
    add_list = [f"{name}={_env_value_text(value)}{newline}"
                for name, value in set_dict.items() if name not in write_set]
    if add_list and text and not _line_end(text):
        text += newline
    bom = _BOM_TEXT if file_text.startswith(_BOM_TEXT) else ""
    content = (bom + text + "".join(add_list)).encode("utf-8")
    _write_private_file(path, content)
    return True


def unload_env_file() -> None:
    """Put back as the shell held them the names the .env file read filled.

    Each name of _ENV_FILE_FILL_DICT, and REQUESTS_CA_FILE_ENV when the file filled
    CA_FILE_ENV, takes the value shell_env() gives it, or is unset when it gives none.
    _ENV_FILE_FILL_DICT is emptied, and a later import of this module leaves the file
    unread (hold_env_file).
    """
    shell_dict = shell_env()
    name_set = set(_ENV_FILE_FILL_DICT)
    if CA_FILE_ENV in name_set:
        name_set.add(REQUESTS_CA_FILE_ENV)
    for name in name_set:
        _OWN_FILL_DICT.pop(name, None)
        if name in shell_dict:
            os.environ[name] = shell_dict[name]
        else:
            os.environ.pop(name, None)
    _ENV_FILE_FILL_DICT.clear()
    if ENV_FILE_PATH:
        hold_env_file(ENV_FILE_PATH)


# The settings that take an empty value although their default is not empty, by the end
# of their names: an instruction file (*_GUIDE) set empty is not read, REPORT_MODEL set
# empty calls no model, and a THINK_LEVEL set empty passes nothing to the provider
_EMPTY_VALUE_SUFFIX_TUPLE = ("_GUIDE", "REPORT_MODEL", "THINK_LEVEL")

# The highest value a numeric setting takes, and the highest value a setting that is a
# number of seconds takes (high_of)
NUMBER_MAX = 2_000_000_000
SECOND_MAX = 2_000_000

# The end of the name of a setting that is a number of seconds
_SECOND_SUFFIX = "_SEC"

# How many characters of a value a SettingError writes out
_MAX_SHOW_VALUE_CHAR = 60


# The code points os.fsdecode() gives the bytes of a value that are not UTF-8, the
# first of them, and what is taken off one to get its byte
_NO_UTF8_FIRST = 0xDC80
_NO_UTF8_LAST = 0xDCFF
_NO_UTF8_BASE = 0xDC00


def _escape_text(char: str) -> str:
    """Return one character that is not printable as the escape it is written out as.

    Examples:
        a line feed                  -> "\\x0a"
        U+3000, an ideographic space -> "\\u3000"
        U+DCFF, the byte 0xFF        -> "\\xff"

    Args:
        char: The character.

    Returns:
        A backslash, "x" and two hex digits up to U+00FF and for a byte that is not
        UTF-8; "u" and four hex digits, or "U" and eight, above.
    """
    code = ord(char)
    if _NO_UTF8_FIRST <= code <= _NO_UTF8_LAST:
        code -= _NO_UTF8_BASE
    if code <= 0xFF:
        return f"\\x{code:02x}"
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


def _value_text(raw: str) -> str:
    """Return a setting's value as a SettingError writes it out, on one line.

    Examples:
        "12abc"                                 -> "12abc"
        "1", an escape, "[2J", a line feed, "2" -> "1\\x1b[2J\\x0a2"

    Args:
        raw: The value as the environment holds it.

    Returns:
        The first _MAX_SHOW_VALUE_CHAR characters of raw, each one that is not printable
        (a control character, a line break, a space other than " ", a byte that is not
        UTF-8) written as _escape_text() gives it, and how long raw is when it is
        longer.
    """
    text = "".join(char if char.isprintable() else _escape_text(char)
                   for char in raw[:_MAX_SHOW_VALUE_CHAR])
    if len(raw) <= _MAX_SHOW_VALUE_CHAR:
        return text
    return f"{text}... ({len(raw)} characters)"


def high_of(name: str) -> int:
    """Return the highest value a numeric setting takes.

    Args:
        name: The setting's name.

    Returns:
        SECOND_MAX for a setting whose name ends in _SECOND_SUFFIX, NUMBER_MAX for any
        other.
    """
    return SECOND_MAX if name.endswith(_SECOND_SUFFIX) else NUMBER_MAX


def second_within_max(second: int | float, source: str) -> int | float:
    """Return a number of seconds worked out from settings, refusing one over SECOND_MAX.

    Args:
        second: The number of seconds.
        source: The settings it was worked out from, as the message names them.

    Returns:
        second.

    Raises:
        SettingError: Naming the settings, when second is over SECOND_MAX.
    """
    if second > SECOND_MAX:
        raise SettingError(f"{source} is {_value_text(str(second))} seconds, and must "
                           f"be {SECOND_MAX} or less")
    return second


def env(name: str, default: str | int | float | bool,
        low: float | None = None) -> str | int | float | bool:
    """Read environment variable name, converted to default's type; default if unset.

    An empty value is the empty string for a setting whose default is the empty string
    and for one whose name ends in _EMPTY_VALUE_SUFFIX_TUPLE. Any other setting, every
    number and true / false among them, takes its default for an empty value.

    Examples:
        env("DEF_K", 12, low=1)         -> int, 1 or more
        env("INCLUDE_CONSTANT", False) -> bool ("false" / "0" / "no" / "off" too)
        env("EDIT_GUIDE", "CODETWINE_LOOM.md") with EDIT_GUIDE="" -> ""

    Args:
        name: The environment variable's name, the same as the setting's name.
        default: The default written in this file. Its type is the conversion target.
        low: The lowest value a numeric setting takes.

    Returns:
        The value converted to default's type.

    Raises:
        SettingError: Naming the setting, when a boolean setting holds something other
            than true / false, or a numeric setting holds what is not a number of its
            type, a number below low, a number above high_of(name), or a number that
            is not finite.
    """
    _ENV_NAME_SET.add(name)
    _DEFAULT_DICT[name] = default
    if low is not None:
        _LOW_DICT[name] = low
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw == "":
        return "" if default == "" or name.endswith(_EMPTY_VALUE_SUFFIX_TUPLE) \
            else default
    if isinstance(default, bool):
        # bool derives from int, so it is checked before int
        raw_lower = raw.strip().lower()
        if raw_lower in ("1", "true", "yes", "on"):
            return True
        if raw_lower in ("0", "false", "no", "off"):
            return False
        raise SettingError(f"{name} must be true / false: {_value_text(raw)}")
    if not isinstance(default, (int, float)):
        return raw
    try:
        value = int(raw) if isinstance(default, int) else float(raw)
    except ValueError:
        what = "a whole number" if isinstance(default, int) else "a number"
        raise SettingError(f"{name} must be {what}: {_value_text(raw)}") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise SettingError(f"{name} must be a finite number: {_value_text(raw)}")
    if low is not None and value < low:
        raise SettingError(f"{name} must be {low} or more: {_value_text(raw)}")
    if value > high_of(name):
        raise SettingError(f"{name} must be {high_of(name)} or less: {_value_text(raw)}")
    return value


def default_of(name: str) -> str | int | float | bool | None:
    """Return the default written in this file for a setting env() read.

    Args:
        name: The setting's name.

    Returns:
        The default, or None for a setting env() has not read.
    """
    return _DEFAULT_DICT.get(name)


def low_of(name: str) -> int | float | None:
    """Return the lowest value a numeric setting env() read takes.

    Args:
        name: The setting's name.

    Returns:
        The lowest value, or None for a setting env() was given none for.
    """
    return _LOW_DICT.get(name)


def json_env(name: str) -> dict[str, str]:
    """Read environment variable name as a JSON object of strings; {} when unset.

    Examples:
        SMOKE_ENV='{"SOME_FEATURE": "off"}' -> {"SOME_FEATURE": "off"}

    Args:
        name: The environment variable's name, the same as the setting's name.

    Returns:
        The parsed object, or {} when the variable is unset or empty.

    Raises:
        SettingError: When the variable holds something that is not a JSON object, or
            an object with a value that is not a string.
    """
    _ENV_NAME_SET.add(name)
    raw = os.environ.get(name)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as e:
        raise SettingError(f"{name} must be a JSON object: {e}") from e
    if not isinstance(value, dict):
        raise SettingError(f"{name} must be a JSON object, not {type(value).__name__}")
    for key, item in value.items():
        if not isinstance(item, str):
            raise SettingError(f"{name} must hold a string under every name: "
                               f"'{_value_text(key)}' holds {type(item).__name__}")
    return value


def str_env(name: str) -> str:
    """Read environment variable name as a string, with no type conversion.

    An unset variable and an empty one both give "".

    Args:
        name: The environment variable's name, the same as the setting's name.

    Returns:
        The variable's value, or "".
    """
    _ENV_NAME_SET.add(name)
    return os.environ.get(name, "")


# The settings set_primary_ext() rebuilds that tell how the tests are run, and those
# that tell how the program is run and checked
TEST_SETTING_TUPLE = ("TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE")
RUN_SETTING_TUPLE = ("RUN_COMMAND", "CHECK_COMMAND")


def hand_env_list(name_tuple: tuple[str, ...]) -> list[str]:
    """Return the names among some whose environment variable holds a value.

    Args:
        name_tuple: Setting names, the same as their environment variables' names.

    Returns:
        The names whose variable holds something besides white space, in the order given.
    """
    return [name for name in name_tuple if str_env(name).strip()]


def shell_list_env(name: str) -> list[str]:
    """Read environment variable name as a shell-quoted list.

    Examples:
        BUILD_EXCLUDE="vendor 'old code'" -> ["vendor", "old code"]
        TEST_COMMAND="make test"          -> ["make", "test"]

    Args:
        name: The environment variable's name, the same as the setting's name.

    Returns:
        The items shlex.split gives, or [] when the variable is unset or empty.

    Raises:
        SettingError: Naming the setting, when shlex.split cannot read the value, as
            with a quotation that is not closed.
    """
    raw = str_env(name)
    try:
        return shlex.split(raw)
    except ValueError as e:
        raise SettingError(f"{name} must be a shell-quoted list ({e}): "
                           f"{_value_text(raw)}") from None


def choice_of(name: str, value: str, choice_tuple: tuple[str, ...]) -> str:
    """Return value as the one of choice_tuple it names, ignoring case and blanks around.

    Examples:
        choice_of("PRESET", " High", ("low", "medium", "high")) -> "high"
        choice_of("AGENT_LOG_LEVEL", "debug", ("DEBUG", "INFO")) -> "DEBUG"

    Args:
        name: The setting's name, which the error names.
        value: The value as it was written.
        choice_tuple: The values the setting takes, as they are returned.

    Returns:
        The member of choice_tuple value names.

    Raises:
        SettingError: When value names none of them.
    """
    choice_dict = {choice.lower(): choice for choice in choice_tuple}
    key = value.strip().lower()
    if key not in choice_dict:
        choice_text = " / ".join(choice or "(empty)" for choice in choice_tuple)
        raise SettingError(f"{name} must be one of {choice_text}: {_value_text(value)}")
    return choice_dict[key]


def choice_env(name: str, default: str, choice_tuple: tuple[str, ...]) -> str:
    """Read environment variable name as one of choice_tuple, as env() reads it.

    Args:
        name: The environment variable's name, the same as the setting's name.
        default: The value when it is unset, or empty and not a setting that takes "".
        choice_tuple: The values the setting takes.

    Returns:
        The member of choice_tuple the variable names (choice_of()).

    Raises:
        SettingError: Naming the setting, when the value is none of choice_tuple.
    """
    _CHOICE_DICT[name] = choice_tuple
    return choice_of(name, env(name, default), choice_tuple)


# == How hard a run tries ======================================
# PRESET picks one column of this table: low, medium or high. A setting named in it
# takes the column's value when it is not set by name; a value set by name wins. The
# table holds how much thinking a call asks for, how many calls and tries a run gets,
# and whether the planning stage runs. How many turns a stage gets is not in it, and
# neither is which model is called (MEDIUM_MODEL and HIGH_MODEL). Every column names the
# same settings, and every name here is read with the column's value as its default
# (preset_env() below, and choice_env() for THINK_LEVEL).
#
# PRESET is not the model's effort. The effort a call asks the model for is THINK_LEVEL,
# which is one of the settings the column sets: "off", "low" and "medium" for the
# columns low, medium and high
_PRESET_TABLE_DICT = {
    "low": {
        "THINK_LEVEL": "off", "AGENT_MAX_TOKEN": 16384,
        "EXPLORE_MAX_CALL": 3, "SEARCH_AGAIN_MAX_CALL": 3,
        "WEB_SEARCH_MAX_CALL": 10, "WEB_FETCH_MAX_CALL": 5, "RUN_PROJECT_MAX_CALL": 10,
        "MAX_PATCH_ROUND": 3, "TWO_STAGE": False,
        "DISPATCH_MAX_TASK": 5, "DISPATCH_MAX_RETRY": 2, "DISPATCH_MAX_RERUN": 1,
        "INTEGRATE_MAX_REFUSAL": 1, "INTEGRATE_MAX_RERUN": 0,
        "TEST_CREATE_MAX_RERUN": 1, "REFACTOR_MAX_RERUN": 1},
    "medium": {
        "THINK_LEVEL": "low", "AGENT_MAX_TOKEN": 32768,
        "EXPLORE_MAX_CALL": 4, "SEARCH_AGAIN_MAX_CALL": 5,
        "WEB_SEARCH_MAX_CALL": 15, "WEB_FETCH_MAX_CALL": 8, "RUN_PROJECT_MAX_CALL": 15,
        "MAX_PATCH_ROUND": 5, "TWO_STAGE": True,
        "DISPATCH_MAX_TASK": 8, "DISPATCH_MAX_RETRY": 3, "DISPATCH_MAX_RERUN": 2,
        "INTEGRATE_MAX_REFUSAL": 2, "INTEGRATE_MAX_RERUN": 1,
        "TEST_CREATE_MAX_RERUN": 2, "REFACTOR_MAX_RERUN": 2},
    "high": {
        "THINK_LEVEL": "medium", "AGENT_MAX_TOKEN": 32768,
        "EXPLORE_MAX_CALL": 5, "SEARCH_AGAIN_MAX_CALL": 10,
        "WEB_SEARCH_MAX_CALL": 20, "WEB_FETCH_MAX_CALL": 10, "RUN_PROJECT_MAX_CALL": 20,
        "MAX_PATCH_ROUND": 8, "TWO_STAGE": True,
        "DISPATCH_MAX_TASK": 10, "DISPATCH_MAX_RETRY": 5, "DISPATCH_MAX_RERUN": 3,
        "INTEGRATE_MAX_REFUSAL": 3, "INTEGRATE_MAX_RERUN": 2,
        "TEST_CREATE_MAX_RERUN": 3, "REFACTOR_MAX_RERUN": 3},
}

# Which column applies: low, medium or high
PRESET_LEVEL_TUPLE = tuple(_PRESET_TABLE_DICT)
PRESET = choice_env("PRESET", "medium", PRESET_LEVEL_TUPLE)
_PRESET_COLUMN_DICT = _PRESET_TABLE_DICT[PRESET]


def preset_env(name: str, low: float | None = None) -> str | int | float | bool:
    """Read environment variable name; the PRESET column's value when unset.

    Args:
        name: The environment variable's name, the same as the setting's name. It has
            to be one _PRESET_TABLE_DICT lists.
        low: The lowest value a numeric setting takes, as env() takes it.

    Returns:
        The value converted to the column value's type, or the column value.

    Raises:
        KeyError: When name is not in the table.
        SettingError: As env() raises it.
    """
    return env(name, _PRESET_COLUMN_DICT[name], low)


# == Paths ====================================================
# --project and --output-dir, which default to these, take their place at start-up.
# PROJECT_DIR defaults to an empty string, which is reported as a missing directory
PROJECT_DIR = env("PROJECT_DIR", "")
OUTPUT_DIR = env("OUTPUT_DIR", "./out")

# The parallelism passed to process_all_files in step 0
CODETWINE_MAX_WORKER = env("CODETWINE_MAX_WORKER", 4, low=1)

# Directory and file names kept out of the analysis, on top of codetwine's own list, as
# a shell-quoted list (BUILD_EXCLUDE="vendor migrations"). An entry holding a / is a
# path from the repository's root, which keeps out that file or everything under that
# directory (BUILD_EXCLUDE="apps/old ./vendor"). --exclude adds to it. It reaches
# codetwine only when codetwine runs: an existing knowledge file keeps what it was made
# with until it is made again
BUILD_EXCLUDE = shell_list_env("BUILD_EXCLUDE")

# The encodings a file of the target repository is decoded with when it has no BOM and
# is not UTF-8, tried in order, as Python codec names separated by commas
# (SOURCE_ENCODING="cp1252"). A file none of them decodes is decoded with the
# encoding codetwine detects. They are handed to codetwine here, and a name that is
# unknown, or is no encoding of text (one that neither decodes bytes nor encodes a
# string), stops the import
SOURCE_ENCODING = [name.strip() for name in str_env("SOURCE_ENCODING").split(",")
                   if name.strip()]
for _encoding in SOURCE_ENCODING:
    try:
        codecs.lookup(_encoding)
        b"".decode(_encoding)
        "".encode(_encoding)
    except (LookupError, ValueError):
        raise SettingError(
            f"SOURCE_ENCODING names an encoding Python does not know: "
            f"{_value_text(_encoding)}. Use Python codec names such as cp1252 "
            f"or latin-1, with commas between them") from None
codetwine_file_utils.SOURCE_ENCODING = SOURCE_ENCODING

# == The models ===============================================
# Two parents, and every stage's own setting follows one of them when unset. The model
# the stages that do not decide what is done call (litellm format): the editing agent
# (EDIT_MODEL), the answering agent (ANSWER_MODEL), the explorer (EXPLORE_MODEL), the
# web search (WEB_SEARCH_MODEL), search step 1 (QUERY_EXPAND_MODEL), the report
# (REPORT_MODEL), the stage that writes tests (TEST_CREATE_MODEL), the stages that
# refactor (REFACTOR_MODEL) and the design documents (DOC_CREATE_MODEL)
MEDIUM_MODEL = env("MEDIUM_MODEL", "anthropic/claude-sonnet-5-5")

# The model the stages that decide what is done call (litellm format): the planning
# stage (PLAN_MODEL), the splitting stage (DISPATCH_MODEL) and the merging stages
# (INTEGRATE_MODEL)
HIGH_MODEL = env("HIGH_MODEL", "anthropic/claude-opus-5-5")


# == Index building and search ================================
# How many definitions search step 2 keeps
DEF_K = env("DEF_K", 12, low=1)

# Whether constant and variable declarations take part in step 2 (see LANG_DICT in
# languages.py)
INCLUDE_CONSTANT = env("INCLUDE_CONSTANT", False)

# How many hops step 3 follows. 0 disables the expansion
EXPAND_DEPTH = 1

# The factor applied to the score per hop. The score a hop hands a definition is
# then divided by how many definitions call it
EXPAND_DECAY = env("EXPAND_DECAY", 0.5, low=0)

# The cap on how many files search_scope() returns
SCOPE_FILE_K = env("SCOPE_FILE_K", 10, low=1)

# The maximum length of one file summary in scope["file_summaries"]
SUMMARY_MAX_CHAR = env("SUMMARY_MAX_CHAR", 4000, low=0)

# How many entries come from the file index (index.file). Merged with the
# definition-derived ranking via RRF
FILE_K = 8

# The RRF (rank fusion) constant. The smaller it is, the heavier rank 1 weighs. The two
# indexes are merged by rank. This one applies to the definition-derived ranking (the
# file index uses RRF_K_FILE)
RRF_K = env("RRF_K", 60, low=0)

# The RRF constant applied to the file index's (index.file) ranks. Below RRF_K the file
# index's top entries come first, above it the definition-derived ranking does. The
# same value as RRF_K weighs the file index's first entry as the definition-derived
# first entry
RRF_K_FILE = env("RRF_K_FILE", 60, low=0)

# How many times a path's words are repeated in the searchable text. 0 leaves paths out
# of the search
PATH_WEIGHT = env("PATH_WEIGHT", 3, low=0)

# The cap on search_again calls. Each stage handed the tool counts its calls from 0,
# except that the planning stage and the explorers of the editing stage after it share
# one count. 0 hands the tool to no stage and takes it out of the explorer's
# instructions
SEARCH_AGAIN_MAX_CALL = preset_env("SEARCH_AGAIN_MAX_CALL", low=0)

# When the number of distinct query terms hitting the top entry point is below this,
# confidence becomes "low". A term more than half the documents carry is not counted
CONFIDENCE_MIN_TERM = 3

# When the top entry point's raw BM25 score is below this it becomes "low". raw is
# divided by log(1 + n_docs) and by hit_terms (at least 1) in bm25_rank, and moves with
# the index's avg_len
CONFIDENCE_MIN_SCORE = env("CONFIDENCE_MIN_SCORE", 0.3, low=0)

# How much term frequency weighs in BM25
BM25_K1 = 1.5

# How much document length weighs in BM25
BM25_B = 0.75

# Whether to run search step 1 (rewriting the request into English search terms)
QUERY_EXPAND = env("QUERY_EXPAND", True)

# Whether search step 1 is handed the repository's overview (overview.py). Outside a
# container a missing or outdated overview is made first; inside one an outdated one is
# not used. Nothing is handed while QUERY_EXPAND is off. --no-overview turns it off
OVERVIEW = env("OVERVIEW", True)

# The model step 1 calls (litellm format). Left unset it follows MEDIUM_MODEL
QUERY_EXPAND_MODEL = env("QUERY_EXPAND_MODEL", MEDIUM_MODEL)

# Step 1's output limit (tokens)
QUERY_EXPAND_MAX_TOKEN = env("QUERY_EXPAND_MAX_TOKEN", 200, low=1)

# How many words step 1 is asked for, as a range written the way the prompt says it.
# QUERY_EXPAND_MAX_TOKEN has to hold whatever this asks for
QUERY_EXPAND_WORD_RANGE = env("QUERY_EXPAND_WORD_RANGE", "20-40")

# == The agent ================================================
# The editing agent's model (litellm format). Left unset it follows MEDIUM_MODEL
EDIT_MODEL = env("EDIT_MODEL", MEDIUM_MODEL)

# The answering agent's model (litellm format): the agent that answers a question
# about the code, on its own or as one of a run's agents. Left unset it follows
# MEDIUM_MODEL
ANSWER_MODEL = env("ANSWER_MODEL", MEDIUM_MODEL)

# The RLM's output limit per call (tokens). It covers thinking and reply together
AGENT_MAX_TOKEN = preset_env("AGENT_MAX_TOKEN", low=1)

# What THINK_LEVEL and every *_THINK_LEVEL may be set to. "" passes nothing, "off" turns
# thinking off, and the three levels ask for as much thinking as that effort calls for.
# A value outside it stops the import, naming the setting (choice_env())
THINK_LEVEL_TUPLE = ("", "off", "low", "medium", "high")

# How much thinking one call asks for: "off" turns thinking off, and "low" / "medium" /
# "high" ask for adaptive thinking at that effort. Empty leaves it to the provider.
# Unset it is the PRESET column's value. The thinking is counted in AGENT_MAX_TOKEN with
# the reply. It is the editing stage's own value, and the default of the other
# *_THINK_LEVEL settings; search step 1 calls with thinking off.
#
# A model that refuses adaptive thinking is given a thinking budget of the output
# limit's 1/8, 1/4 or 1/2 for low, medium or high (at least 1024 tokens), and one that
# refuses that too is called without either. A form a model has refused is not sent to
# it again in the same process. A model that cannot stop thinking (litellm's
# thinking_always_on) is asked for "low" where "off" is set
THINK_LEVEL = choice_env("THINK_LEVEL", _PRESET_COLUMN_DICT["THINK_LEVEL"],
                         THINK_LEVEL_TUPLE)

# How many times a turn whose reply was cut off without yielding a single code block is
# asked again, without spending a turn. 0 drops that turn instead
CUT_REPLY_MAX_RETRY = env("CUT_REPLY_MAX_RETRY", 2, low=0)

# The API key. Read from the environment; never written in the source
LLM_API_KEY = env("LLM_API_KEY", "")

# A non-standard endpoint, for Ollama and the like. Empty is not passed to litellm
LLM_API_BASE = env("LLM_API_BASE", "")

# How many times a call is made again after the provider answered it as too busy or as
# failing on its side, its connection failed or an attempt ran out of LLM_CALL_TIMEOUT_SEC
# or LLM_IDLE_TIMEOUT_SEC (providercall.py). Other errors are not retried
LLM_MAX_RETRY = env("LLM_MAX_RETRY", 3, low=0)

# The wait before the first retry of a call (seconds). Each retry after it waits twice
# as long as the one before, or the retry-after the provider sends when that is longer,
# at most LLM_CALL_TIMEOUT_SEC
LLM_RETRY_WAIT_SEC = env("LLM_RETRY_WAIT_SEC", 10, low=0)

# How long one attempt of a call may take, from the request to the end of the reply,
# before it is given up and the next is made (seconds). It covers thinking and reply
# together, and has to be above what AGENT_MAX_TOKEN worth of output takes
LLM_CALL_TIMEOUT_SEC = env("LLM_CALL_TIMEOUT_SEC", 600, low=1)

# How long a turn's reply, received as a stream, may go with no part of it arriving
# before the attempt is given up and the next is made (seconds). It counts from the
# request to the first part and between two parts, thinking included. 0 gives up only at
# LLM_CALL_TIMEOUT_SEC
LLM_IDLE_TIMEOUT_SEC = env("LLM_IDLE_TIMEOUT_SEC", 120, low=0)

# Whether a stage's prompt is sent as blocks with a cache point after the part that does
# not change over a run (the instructions, the variables, the tools and the request) and
# after the newest turn of the history: the provider reads what it already processed
# from its cache. False sends the prompt as one string
PROMPT_CACHE = env("PROMPT_CACHE", True)

# The two forms a broker address takes
BROKER_UNIX_SCHEME = "unix://"
BROKER_TCP_SCHEME = "tcp://"

# The highest port a tcp:// address names
_MAX_PORT = 65535


def broker_address_of(url: str) -> tuple[str, str | tuple[str, int]]:
    """Split a broker address into its scheme and what a socket of that kind is given.

    Examples:
        "unix:///run/ctloom/llm.sock" -> ("unix://", "/run/ctloom/llm.sock")
        "tcp://broker:9000"        -> ("tcp://", ("broker", 9000))

    Args:
        url: The address, as LLM_BROKER holds it.

    Returns:
        (BROKER_UNIX_SCHEME, the socket's path) or (BROKER_TCP_SCHEME, (host, port)).

    Raises:
        SettingError: When url starts with neither scheme, when it names no path, or
            when a tcp:// address has no host, a port that is not ASCII digits from 1
            to _MAX_PORT, or an IPv6 address for its host.
    """
    url_text = _value_text(url)
    if url.startswith(BROKER_UNIX_SCHEME):
        path = url[len(BROKER_UNIX_SCHEME):]
        if not path:
            raise SettingError(f"LLM_BROKER={url_text} names no socket path. Write it as "
                               f"{BROKER_UNIX_SCHEME}/path/to/llm.sock")
        return BROKER_UNIX_SCHEME, path
    if url.startswith(BROKER_TCP_SCHEME):
        host, _, port = url[len(BROKER_TCP_SCHEME):].rpartition(":")
        if not host or not (port.isascii() and port.isdigit()):
            raise SettingError(f"LLM_BROKER={url_text} names no host and port. Write it as "
                               f"{BROKER_TCP_SCHEME}<host>:<port>")
        # The digits are counted before they are read as a number
        digit_text = port.lstrip("0") or "0"
        if len(digit_text) > len(str(_MAX_PORT)) or not 1 <= int(digit_text) <= _MAX_PORT:
            raise SettingError(f"LLM_BROKER={url_text} names a port outside 1 to {_MAX_PORT}")
        if ":" in host or host.startswith("["):
            raise SettingError(f"LLM_BROKER={url_text} names an IPv6 address, which a "
                               f"{BROKER_TCP_SCHEME} address does not take. Write the "
                               f"host as a name or an IPv4 address")
        return BROKER_TCP_SCHEME, (host, int(digit_text))
    raise SettingError(f"LLM_BROKER={url_text} starts with neither {BROKER_UNIX_SCHEME} nor "
                       f"{BROKER_TCP_SCHEME}")


def absolute_broker_url(url: str) -> str:
    """Return a broker address with the path of a Unix socket made absolute.

    Examples:
        "unix://run/llm.sock" -> "unix:///<working directory>/run/llm.sock"
        "unix:///run/ctloom/llm.sock" -> "unix:///run/ctloom/llm.sock"
        "tcp://broker:9000"   -> "tcp://broker:9000"

    Args:
        url: The address, as LLM_BROKER holds it.

    Returns:
        The address, a relative socket path in it read from the working directory.

    Raises:
        SettingError: What broker_address_of() raises.
    """
    scheme, address = broker_address_of(url)
    return scheme + os.path.abspath(address) if scheme == BROKER_UNIX_SCHEME else url


# Where the call is made. Empty calls litellm in this process, with LLM_API_KEY and
# LLM_API_BASE read here. A value sends the call to a broker instead, which holds the
# credentials and calls litellm itself, and neither of those two settings is read or
# sent. It takes one of two forms:
#   unix:///run/ctloom/llm.sock   a Unix socket, for a broker beside the container
#   tcp://broker:9000          a TCP address, for a broker somewhere else, its host a
#                              name or an IPv4 address and its port 1 to 65535
# A value of neither form stops the import (broker_address_of()). A socket path that is
# not absolute is read from the directory the command is run from, and held here as an
# absolute path
LLM_BROKER = env("LLM_BROKER", "")
if LLM_BROKER:
    LLM_BROKER = absolute_broker_url(LLM_BROKER)

# How long one call through the broker is waited for (seconds): every attempt of it,
# LLM_CALL_TIMEOUT_SEC each, and the waits between them, at most LLM_CALL_TIMEOUT_SEC
# each. A search is waited for 2 x WEB_SEARCH_TIMEOUT_SEC longer and a page read
# 2 x WEB_FETCH_TIMEOUT_SEC longer. It is also how long the broker waits between two
# requests on one connection before closing it
LLM_BROKER_TIMEOUT_SEC = second_within_max(
    LLM_CALL_TIMEOUT_SEC * (2 * LLM_MAX_RETRY + 1),
    "LLM_CALL_TIMEOUT_SEC x (2 x LLM_MAX_RETRY + 1)")

# How long the sandbox waits for a broker it started to be listening, and then for it to
# end when it is stopped (seconds)
BROKER_START_TIMEOUT_SEC = env("BROKER_START_TIMEOUT_SEC", 10, low=1)

# The largest request the broker reads as one line (bytes). A connection that sends
# more than it is answered with an error and closed
LLM_BROKER_MAX_REQUEST_BYTE = env("LLM_BROKER_MAX_REQUEST_BYTE", 8388608, low=1)

# How many callers one broker answers at once. It is the socket's backlog and the cap on
# the threads serving connections alike; a connection over it is refused
LLM_BROKER_MAX_CONNECTION = env("LLM_BROKER_MAX_CONNECTION", 16, low=1)

# How many model calls one broker may make, counted by the broker. 0 sets no limit
LLM_MAX_CALL = env("LLM_MAX_CALL", 0, low=0)

# The loop's limit on turns: the editing agent's, and that of a loop given none of its
# own
RLM_MAX_TURN = env("RLM_MAX_TURN", 40, low=1)

# How many characters of a reply, a line or an entry are quoted where it is shown in
# part
RAW_EXCERPT_CHAR = 2000

# How much of a failure is quoted for the operator to read. No part of it reaches a
# prompt
LOG_EXCERPT_CHAR = 200

# How much of one turn's output is kept in the history. 0 keeps the whole output
RLM_MAX_OUTPUT_CHAR = env("RLM_MAX_OUTPUT_CHAR", 40000, low=0)

# Whether the editing agent, when it used every turn without COMPLETE(), is asked once,
# in a call that runs no code, for what it changed. The reply becomes the answer. The
# same shape as PLAN_LATE_ANSWER
EDIT_LATE_ANSWER = env("EDIT_LATE_ANSWER", True)

# Over the last this many turns an agent is told to hand over what it has with
# COMPLETE(), in the note rlm.make_deadline_note() builds. 0 turns the note off
EDIT_DEADLINE_TURN = env("EDIT_DEADLINE_TURN", 3, low=0)

# == The repository's own instruction file ====================
# The file that says how this repository is to be changed: its conventions, what has to
# be run before a change is done, what must not be touched. It is the operator's, not
# the model's, and it is followed by the planning stage and the editing stage.
#
# The default is this tool's own file. A repository that wants another file read names
# it: EDIT_GUIDE=AGENTS.md, or edit --guide AGENTS.md.
#
# A relative path is resolved against PROJECT_DIR: the default is "the file in the
# repository being changed", and an absolute path names a file kept outside any one
# repository. "" turns it off, and a repository that has no such file adds nothing to
# any prompt
EDIT_GUIDE = env("EDIT_GUIDE", "CODETWINE_LOOM.md")

# The cap on how much of that file reaches the model. It sits in the instructions of
# those two stages. Over the cap the rest is dropped with the same marker a cut turn
# output carries. 0 means no cap
EDIT_GUIDE_MAX_CHAR = env("EDIT_GUIDE_MAX_CHAR", 40000, low=0)

# Each of the stages that follow a merge has an instruction file of its own, as those
# two stages have EDIT_GUIDE, and a stage is handed its own file alone. Each path is
# resolved as EDIT_GUIDE's is and cut to the same cap, "" turns it off, and a repository
# that has no such file adds nothing to any prompt.
#
# The file that says how this repository's tests are to be written: the framework, the
# fixtures and helpers, the naming, what a test must not do. The stage that writes the
# tests of a combined change follows it
TEST_GUIDE = env("TEST_GUIDE", "CODETWINE_LOOM_TEST.md")

# The file that says how this repository is to be refactored: what counts as worth
# tidying here and what is left alone. The stage that refactors the change follows it
REFACTOR_GUIDE = env("REFACTOR_GUIDE", "CODETWINE_LOOM_REFACTOR.md")

# The file that says how this repository's tests are to be refactored. The stage that
# refactors the tests follows it
TEST_REFACTOR_GUIDE = env("TEST_REFACTOR_GUIDE", "CODETWINE_LOOM_TEST_REFACTOR.md")

# == Splitting one request and handing it out =================
# The model the splitting stage (dispatch) calls. It reads the whole request once and
# decides what every agent is asked - the questions they answer or the changes they
# make. Left unset it follows HIGH_MODEL
DISPATCH_MODEL = env("DISPATCH_MODEL", HIGH_MODEL)

# The splitting stage's output limit per call (tokens) and how much thinking it asks
# for, apart from the editing stage's. Left unset they follow AGENT_MAX_TOKEN and
# THINK_LEVEL
DISPATCH_MAX_TOKEN = env("DISPATCH_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
DISPATCH_THINK_LEVEL = choice_env("DISPATCH_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)

# The answering agent's own settings, apart from the editing stage's. Left unset they
# follow the editing stage's
ANSWER_MAX_TOKEN = env("ANSWER_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
ANSWER_THINK_LEVEL = choice_env("ANSWER_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)
ANSWER_MAX_TURN = env("ANSWER_MAX_TURN", RLM_MAX_TURN, low=1)
ANSWER_DEADLINE_TURN = env("ANSWER_DEADLINE_TURN", EDIT_DEADLINE_TURN, low=0)
ANSWER_LATE_ANSWER = env("ANSWER_LATE_ANSWER", EDIT_LATE_ANSWER)

# How many turns the splitting stage (dispatch) gets. It reads and runs but never edits.
# Left unset it follows RLM_MAX_TURN
DISPATCH_MAX_TURN = env("DISPATCH_MAX_TURN", RLM_MAX_TURN, low=1)

# How long an answer may be: each answering agent's, and the combined one the user
# reads. The instructions ask for it; an answer over it is sent back once to be
# shortened, and cut at the end only if it is still over after that. 0 means no cap
DISPATCH_ANSWER_MAX_CHAR = env("DISPATCH_ANSWER_MAX_CHAR", 6000, low=0)

# Over the last this many turns the stage is told to hand over what it has. 0 turns the
# note off. It is PLAN_DEADLINE_TURN's counterpart. Left unset it follows
# EDIT_DEADLINE_TURN
DISPATCH_DEADLINE_TURN = env("DISPATCH_DEADLINE_TURN", EDIT_DEADLINE_TURN, low=0)

# How many tasks, or questions, one request may be split into. Every one is an agent of
# its own
DISPATCH_MAX_TASK = preset_env("DISPATCH_MAX_TASK", low=1)

# How long one task's request, or one question, may be. It is the whole of what the
# agent taking it is told. An entry over it is sent back to the splitting stage once to
# be shortened, and cut at the end only if it is still over after that. 0 means no cap
DISPATCH_TASK_MAX_CHAR = env("DISPATCH_TASK_MAX_CHAR", 4000, low=0)

# How many things the splitting stage may hand back for the person who asked to settle,
# in place of a split. The run that gets them starts no agent. 0 takes that way out
# away, and the instructions do not mention it
DISPATCH_MAX_UNCLEAR = env("DISPATCH_MAX_UNCLEAR", 3, low=0)

# How many agents run at once when a run hands out its tasks, its questions or its test
# tasks. The budgets and the memory limit are per agent
DISPATCH_PARALLEL = env("DISPATCH_PARALLEL", 3, low=1)

# How many more times a task whose agent did not finish is handed out, each time to an
# agent on a branch of its own. 0 hands it out once and no more. A task still unfinished
# after the last try leaves nothing combined
DISPATCH_MAX_RETRY = preset_env("DISPATCH_MAX_RETRY", low=0)

# How many more times a request whose dispatching run did not get through is run again,
# each time carrying on from the branches the run before it left, as passing them with
# --resume does. 0 runs it once and no more. Every run again is a whole dispatching run:
# the reading run, an agent per task and the merging run
DISPATCH_MAX_RERUN = preset_env("DISPATCH_MAX_RERUN", low=0)

# How many runs' leftovers are kept. A dispatching run makes a branch per task and an
# output directory per agent. At the end of a dispatching run everything older than the
# newest few is taken away: the branches of the tasks, and the turn logs, patches and
# records beside them. What a run produced - the branch a union landed on - is not
# touched, and neither is a branch named in --resume. 0 takes nothing away
DISPATCH_KEEP_RUN = env("DISPATCH_KEEP_RUN", 3, low=0)

# How many tokens, input and output added up, the model calls of one run, sandbox or
# integrate command may use across every container it starts and the overview and the
# report it makes. Once the count reaches it no further container is started. A
# container is given what is left of it when it starts, and its model calls are refused
# once they have used that many. 0 sets no limit
MAX_TOKEN_PER_RUN = env("MAX_TOKEN_PER_RUN", 0, low=0)

# The model that says, at the end of a dispatching run, what was changed and where. It
# is handed the request, the files the change touched and its diff. Empty leaves the
# report to the machine-written file list alone. Left unset it follows MEDIUM_MODEL
REPORT_MODEL = env("REPORT_MODEL", MEDIUM_MODEL)

# The report's output limit for that one call (tokens). Left unset it follows
# AGENT_MAX_TOKEN
REPORT_MAX_TOKEN = env("REPORT_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)

# How much of the change's diff the report's model is handed (characters). 0 means no cap
REPORT_DIFF_MAX_CHAR = env("REPORT_DIFF_MAX_CHAR", 60000, low=0)

# == Combining what several agents did ========================
# The model the stages that combine what several agents did call: the agents' branches
# into one, or their answers to a question into one. Left unset it follows HIGH_MODEL
INTEGRATE_MODEL = env("INTEGRATE_MODEL", HIGH_MODEL)

# Those stages' output limit per call (tokens) and how much thinking they ask for.
# Left unset they follow AGENT_MAX_TOKEN and THINK_LEVEL
INTEGRATE_MAX_TOKEN = env("INTEGRATE_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
INTEGRATE_THINK_LEVEL = choice_env("INTEGRATE_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)

# How many turns those stages get. The merging stage is handed what the merge could not
# settle and the combining stage of a question what the agents answered. A merge still
# unsettled after it is run again INTEGRATE_MAX_RERUN times more
INTEGRATE_MAX_TURN = env("INTEGRATE_MAX_TURN", 50, low=1)

# Over the last this many turns those stages are told to settle what is left and
# finish. 0 turns the note off. Left unset it follows EDIT_DEADLINE_TURN
INTEGRATE_DEADLINE_TURN = env("INTEGRATE_DEADLINE_TURN", EDIT_DEADLINE_TURN, low=0)

# How many items each part of the work list carries: conflicted files, broken calls, and
# the names offered in place of one that is gone. A list cut by this sets the work
# list's "complete" to False. The same count bounds the other lists a stage is shown of
# what was read off the branches, and each list of the state block a stage reads before
# every turn
INTEGRATE_WORK_MAX_ITEM = env("INTEGRATE_WORK_MAX_ITEM", 40, low=1)

# How many times the merging stage's COMPLETE() is sent back while the check still finds
# conflict markers, calls that do not resolve or tests the settling broke. 0 accepts
# the first call as it is
INTEGRATE_MAX_REFUSAL = preset_env("INTEGRATE_MAX_REFUSAL", low=0)

# How many more times a dispatching run's merge that the check still finds unsettled is
# run again, each time carrying on from the union the merge before it left. 0 runs it
# once and no more. A merge still unsettled after the last one is what DISPATCH_MAX_RERUN
# runs the whole request again from. A question's combining run is run again the same
# number of times, over the answers the agents already gave, when it hands none back
INTEGRATE_MAX_RERUN = preset_env("INTEGRATE_MAX_RERUN", low=0)


# == Writing tests for a combined change ======================
# Whether a merging run whose merge was settled is followed by a stage that writes tests
# for the definitions the change touched, before the result is put into the working
# tree. The stage runs in a container of its own and its change is a commit of its own
# on top of the merge. A repository with no test command, one with no test file and no
# dependency file the stage could write, and a change that touched no definition outside
# the tests, call no model. A repository whose tests run nothing gets its first tests
# here, and the stage writes the language's dependency files where a test needs a
# package or a test script. --no-test-create turns it off
TEST_CREATE = env("TEST_CREATE", True)

# The model that stage calls (litellm format). Left unset it follows MEDIUM_MODEL
TEST_CREATE_MODEL = env("TEST_CREATE_MODEL", MEDIUM_MODEL)

# That stage's output limit per call (tokens) and how much thinking it asks for. Left
# unset they follow AGENT_MAX_TOKEN and THINK_LEVEL
TEST_CREATE_MAX_TOKEN = env("TEST_CREATE_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
TEST_CREATE_THINK_LEVEL = choice_env("TEST_CREATE_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)

# How many turns each of that stage's runs gets
TEST_CREATE_MAX_TURN = env("TEST_CREATE_MAX_TURN", 50, low=1)

# How many of the definitions the change touched one container of that stage writes
# tests for. A change touching more is split into test tasks of at most this many, a
# file's definitions staying together, and the files that share a test file too unless
# they hold more than this many. Each is written in a container of its own,
# DISPATCH_PARALLEL of them at once; the test tasks that files sharing a test file were
# split into are written one after another, each over what the one before it left. The
# test tasks' tests are then combined into one commit. 0 writes them all in one container
TEST_CREATE_MAX_DEFINITION = env("TEST_CREATE_MAX_DEFINITION", 10, low=0)

# How many times that stage's COMPLETE() is sent back while the check still finds a call
# that does not resolve or a test that fails. 0 accepts the first call as it is. Left
# unset it follows INTEGRATE_MAX_REFUSAL
TEST_CREATE_MAX_REFUSAL = env("TEST_CREATE_MAX_REFUSAL", INTEGRATE_MAX_REFUSAL, low=0)

# How many more times that stage is run when it is left unsettled, each time carrying on
# from what the run before it left, as integrate --resume does: the tests of the test
# tasks that settled are kept and only the files still open are written again. 0 runs
# it once and no more. Unset it is the PRESET column's value
TEST_CREATE_MAX_RERUN = preset_env("TEST_CREATE_MAX_RERUN", low=0)


# == Refactoring a combined change ============================
# Whether a merging run whose merge was settled is followed by a stage that refactors
# the files the change touched outside the tests, before the result is put into the
# working tree. It comes after the stage that writes tests, runs in one container and
# lands as a commit of its own. It writes to no test, and a run that changes nothing
# leaves no commit. A repository whose tests cannot be run, and a change that touched no
# definition outside the tests, call no model. --no-refactor turns it off
REFACTOR = env("REFACTOR", True)

# Whether a stage that refactors the tests the change wrote or changed follows, in the
# same way and last of the three. It writes to nothing outside the tests, and keeps
# every test case it found. --no-test-refactor turns it off
TEST_REFACTOR = env("TEST_REFACTOR", True)

# The model both of those stages call (litellm format). Left unset it follows
# MEDIUM_MODEL
REFACTOR_MODEL = env("REFACTOR_MODEL", MEDIUM_MODEL)

# Their output limit per call (tokens) and how much thinking they ask for. Left unset
# they follow AGENT_MAX_TOKEN and THINK_LEVEL
REFACTOR_MAX_TOKEN = env("REFACTOR_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
REFACTOR_THINK_LEVEL = choice_env("REFACTOR_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)

# How many turns each of them gets
REFACTOR_MAX_TURN = env("REFACTOR_MAX_TURN", 50, low=1)

# How many times their COMPLETE() is sent back while the check still finds something.
# 0 accepts the first call as it is. Left unset it follows INTEGRATE_MAX_REFUSAL
REFACTOR_MAX_REFUSAL = env("REFACTOR_MAX_REFUSAL", INTEGRATE_MAX_REFUSAL, low=0)

# How many more times each of them is run when it is left unsettled, each time over the
# commit it started at, with what it left dropped, as integrate --resume does. 0 runs it
# once and no more. Unset it is the PRESET column's value
REFACTOR_MAX_RERUN = preset_env("REFACTOR_MAX_RERUN", low=0)


# == Carrying one change over to the next run ================
# Whether what an edit did is recorded in <OUTPUT_DIR>/<project>/edit_history.json, and
# the last unfinished entry read back into the next run's request
EDIT_HISTORY = env("EDIT_HISTORY", True)

# How many entries the history file keeps. Older ones are dropped, oldest first
EDIT_HISTORY_KEEP = env("EDIT_HISTORY_KEEP", 5, low=1)

# The caps on one entry
EDIT_HISTORY_REQUEST_CHAR = env("EDIT_HISTORY_REQUEST_CHAR", 400, low=0)
EDIT_HISTORY_DID_CHAR = env("EDIT_HISTORY_DID_CHAR", 800, low=0)
EDIT_HISTORY_MAX_FILE = env("EDIT_HISTORY_MAX_FILE", 30, low=1)

# How many items are named before the rest becomes a count
MAX_NAME_ITEM = 4

# How much of the previous run's last program output is carried over into every prompt
# of the next run
EDIT_HISTORY_LAST_RUN_CHAR = 700

# == Folding the history =====================================
# The budget for the rendered history of one prompt (characters). Above it, the oldest
# turns' output is folded down to HISTORY_FOLD_HEAD_CHAR, oldest first, until the
# history is within half the budget; when the outputs alone do not get it there, the
# code of the same old turns is folded the same way. A turn folded once stays folded
# for the rest of the stage's run, and no further turn is folded until the history,
# measured with those folded, is over the budget again. While it is over the budget with
# every old turn folded, each turn folds the one that has just left the kept turns. 0
# turns the folding off and the whole history goes in. A prompt the provider refuses for
# its length lowers this for the rest of that stage's run, to half of it and to less
# than the history that prompt carried, and is sent once more
HISTORY_MAX_CHAR = env("HISTORY_MAX_CHAR", 600000, low=0)

# How many of the most recent turns are never folded, whatever the budget says
HISTORY_KEEP_TURN = env("HISTORY_KEEP_TURN", 3, low=0)

# How much of the head of a folded turn's output, or code, stays in the history
HISTORY_FOLD_HEAD_CHAR = env("HISTORY_FOLD_HEAD_CHAR", 2000, low=0)


# == Two stages: a planning RLM, then the editing RLM =========
# Whether to run the planning stage before the editing one. False is a single-stage
# run. Unset it is the PRESET column's value, which turns it on at medium and high
TWO_STAGE = preset_env("TWO_STAGE")

# The model the planning stage calls (litellm format). It only reads and writes a plan,
# and the plan is what the editing stage is held to. Left unset it follows HIGH_MODEL
PLAN_MODEL = env("PLAN_MODEL", HIGH_MODEL)

# How many turns the planning stage gets. They cover reading and any run_project call.
# Left unset it follows RLM_MAX_TURN
PLAN_MAX_TURN = env("PLAN_MAX_TURN", RLM_MAX_TURN, low=1)

# How many files one plan may list. The plan's "order" and "interfaces" lists are cut
# to the same count
PLAN_MAX_FILE = env("PLAN_MAX_FILE", 8, low=1)

# The cap on the plan as JSON (characters). The instructions ask for it, and a plan
# over it is trimmed from the end after it is read (plan.fit_to_budget). 0 means no cap
PLAN_ANSWER_MAX_CHAR = env("PLAN_ANSWER_MAX_CHAR", 40000, low=0)

# The planning stage's output limit per call (tokens) and how much thinking it asks for
# (the values of THINK_LEVEL), apart from the editing stage's. Left unset they follow
# AGENT_MAX_TOKEN and THINK_LEVEL
PLAN_MAX_TOKEN = env("PLAN_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
PLAN_THINK_LEVEL = choice_env("PLAN_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)

# How many lines of "what I checked rather than assumed" one plan may carry
PLAN_MAX_FACT = env("PLAN_MAX_FACT", 12, low=1)

# Whether to make one extra call for the plan when the planning stage used every turn
# without COMPLETE. The call runs no code and asks for the JSON alone; False leaves the
# run with no plan at all
PLAN_LATE_ANSWER = env("PLAN_LATE_ANSWER", True)

# Over the last this many turns the planning stage is told to hand over what it has.
# 0 turns the note off. Left unset it follows EDIT_DEADLINE_TURN
PLAN_DEADLINE_TURN = env("PLAN_DEADLINE_TURN", EDIT_DEADLINE_TURN, low=0)

# Whether COMPLETE() in the editing stage is refused once while a file the plan listed
# has not been touched. It is refused at most once per request
PLAN_COMPLETE_GUARD = env("PLAN_COMPLETE_GUARD", True)

# The cap on what read_source_file returns per call (over the whole return value,
# header and continuation note included). A file over the cap comes back up to the last
# line that fits, with a note on how to continue. 0 leaves RLM_MAX_OUTPUT_CHAR as the
# only cap, and whole files come back when that is 0 too
READ_MAX_CHAR = env("READ_MAX_CHAR", 40000, low=0)

# The cap on lines grep_source returns. What actually comes back is the smaller of this
# and "as many as fit in RLM_MAX_OUTPUT_CHAR" (source.hit_limit)
GREP_MAX_HIT = env("GREP_MAX_HIT", 200, low=1)

# The cap on lines counted by grep_source(count_only=True). The line contents are
# dropped and only per-path counts are kept. With count_only=False it also bounds the
# hits counted into "counts"
GREP_MAX_HIT_COUNT = env("GREP_MAX_HIT_COUNT", 5000, low=1)

# How many context lines grep_source adds around its first hits. 0 adds none
GREP_CONTEXT_LINE = env("GREP_CONTEXT_LINE", 3, low=0)

# How many hits get context lines. Later hits return path / line / text only
GREP_CONTEXT_HIT = env("GREP_CONTEXT_HIT", 5, low=0)

# The estimated non-body cost of one printed hit (key names, brackets, quotes). Used to
# work out how many hits fit in RLM_MAX_OUTPUT_CHAR
GREP_HIT_OVERHEAD_CHAR = env("GREP_HIT_OVERHEAD_CHAR", 60, low=1)

# How much of one line grep_source keeps. Applies to the hit line and its context alike,
# and is counted per hit, with GREP_HIT_OVERHEAD_CHAR, when working out how many hits
# fit in RLM_MAX_OUTPUT_CHAR
GREP_LINE_MAX_CHAR = env("GREP_LINE_MAX_CHAR", 200, low=0)

# Directory names grep_source never walks into, comma separated: dependencies and build
# output. The directories the primary language's build_output_tuple (languages.py) names
# by name alone are skipped on top of these; add to it for a repository that keeps its
# build somewhere else
GREP_SKIP_DIR = env(
    "GREP_SKIP_DIR",
    "__pycache__,.git,.github,.venv,venv,.tox,.mypy_cache,.pytest_cache,"
    "node_modules,dist,build,out,target,.gradle,.mvn,vendor,.idea,.vscode")

# == The explorer =============================================
# The model the explorer calls (litellm format). Left unset it follows MEDIUM_MODEL
EXPLORE_MODEL = env("EXPLORE_MODEL", MEDIUM_MODEL)

# One explorer's output limit per call (tokens) and how much thinking it asks for. Left
# unset they follow AGENT_MAX_TOKEN and THINK_LEVEL
EXPLORE_MAX_TOKEN = env("EXPLORE_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)
EXPLORE_THINK_LEVEL = choice_env("EXPLORE_THINK_LEVEL", THINK_LEVEL, THINK_LEVEL_TUPLE)

# How many turns one explorer gets. One that uses them all without COMPLETE() is asked
# once more, in a call that runs nothing, for the places it has found
EXPLORE_MAX_TURN = env("EXPLORE_MAX_TURN", 20, low=1)

# How many explore calls the editing stage gets. 0 disables the explorer
EXPLORE_MAX_CALL = preset_env("EXPLORE_MAX_CALL", low=0)

# The cap on the explorer's findings as JSON (characters). The instructions ask for it,
# and a list over it is trimmed from the end after it is read. 0 means no cap
EXPLORE_ANSWER_MAX_CHAR = env("EXPLORE_ANSWER_MAX_CHAR", 40000, low=0)

# How many findings one explore call returns. It goes into the instructions and is also
# applied to the parsed array
EXPLORE_MAX_RESULT = env("EXPLORE_MAX_RESULT", 6, low=1)

# == Asking the web (a fact that is outside the repository) ===
# The tool answers a fact about something the repository does not contain, such as how
# some other project does a thing today.
#
# The search is one HTTP request made by whichever process holds the network - the
# broker for a container that has none, this process otherwise. No model takes part in
# it, except with WEB_SEARCH_BY_THE_MODEL.
#
# The cap on web_search calls. Each stage handed the tool counts its calls from 0,
# except that the planning stage and the editing stage after it share one count. 0 hands
# the tool to none and takes it out of the instructions.
#
# A budget alone does not turn the tool on: the tool and its instructions appear only
# when there is a way to search as well (websearch.can_search())
WEB_SEARCH_MAX_CALL = preset_env("WEB_SEARCH_MAX_CALL", low=0)

# The search APIs a run can name. Each entry describes that API's endpoint, with {key}
# standing for WEB_SEARCH_API_KEY: "url", "method" (GET puts the parameters in the URL,
# POST sends them as a JSON body), "query_param" (the parameter the question goes in),
# "params" (further parameters, left out where the API needs none), "headers",
# "results_path" (a dotted path to the result list in the JSON reply) and "result_keys"
# (the keys kept from one result, comma separated; "key:name" keeps key under name;
# every entry keeps title, url and content)
WEB_SEARCH_PRESET_DICT = {
    "tavily": {"url": "https://api.tavily.com/search", "method": "POST",
               "query_param": "query",
               "headers": {"Authorization": "Bearer {key}"},
               "results_path": "results", "result_keys": "title,url,content"},
    "brave": {"url": "https://api.search.brave.com/res/v1/web/search", "method": "GET",
              "query_param": "q", "params": {"text_decorations": "false"},
              "headers": {"X-Subscription-Token": "{key}", "Accept": "application/json"},
              "results_path": "web.results", "result_keys": "title,url,description:content"},
    "ollama": {"url": "https://ollama.com/api/web_search", "method": "POST",
               "query_param": "query",
               "headers": {"Authorization": "Bearer {key}"},
               "results_path": "results", "result_keys": "title,url,content"},
}

# The value of WEB_SEARCH_PROVIDER that has the model's own provider search: one call
# of WEB_SEARCH_MODEL with the provider's web search on, made with the LLM credentials.
# The model reads what it finds and answers; the answer comes back as the first result
# and the pages it cited follow
WEB_SEARCH_BY_THE_MODEL = "model"

# Which of WEB_SEARCH_PRESET_DICT to search, or WEB_SEARCH_BY_THE_MODEL. Empty, there is
# no search. It is read by whichever process makes the request: a run with a broker sets
# it, and the key, there and not in the container
WEB_SEARCH_PROVIDER = choice_env("WEB_SEARCH_PROVIDER", "",
                                 ("", *WEB_SEARCH_PRESET_DICT, WEB_SEARCH_BY_THE_MODEL))
_WEB_SEARCH_ENDPOINT_DICT = WEB_SEARCH_PRESET_DICT.get(WEB_SEARCH_PROVIDER, {})

# The key the preset's API is used with. It goes where the preset puts it and stays
# with the process that makes the request
WEB_SEARCH_API_KEY = env("WEB_SEARCH_API_KEY", "")
if _WEB_SEARCH_ENDPOINT_DICT and not WEB_SEARCH_API_KEY:
    raise SettingError(f"WEB_SEARCH_PROVIDER={WEB_SEARCH_PROVIDER} needs its key in "
                       f"WEB_SEARCH_API_KEY")

# The model that searches when WEB_SEARCH_PROVIDER is WEB_SEARCH_BY_THE_MODEL, and the
# most tokens its answer may take. The model follows MEDIUM_MODEL and the tokens
# AGENT_MAX_TOKEN when unset
WEB_SEARCH_MODEL = env("WEB_SEARCH_MODEL", MEDIUM_MODEL)
WEB_SEARCH_MAX_TOKEN = env("WEB_SEARCH_MAX_TOKEN", AGENT_MAX_TOKEN, low=1)

# How much of the pages found that model reads before it answers: "low", "medium" or
# "high", the provider's own scale. What comes back here is cut to
# WEB_SEARCH_RESULT_MAX_CHAR whatever it read
WEB_SEARCH_CONTEXT_SIZE = choice_env("WEB_SEARCH_CONTEXT_SIZE", "low",
                                     ("low", "medium", "high"))

# The endpoint of the named preset, taken from WEB_SEARCH_PRESET_DICT and not read from
# the environment. WEB_SEARCH_URL is empty when no preset is named
WEB_SEARCH_URL = _WEB_SEARCH_ENDPOINT_DICT.get("url", "")
WEB_SEARCH_METHOD = _WEB_SEARCH_ENDPOINT_DICT.get("method", "GET")
WEB_SEARCH_QUERY_PARAM = _WEB_SEARCH_ENDPOINT_DICT.get("query_param", "q")
WEB_SEARCH_PARAM = _WEB_SEARCH_ENDPOINT_DICT.get("params", {})
WEB_SEARCH_HEADER = {name: value.format(key=WEB_SEARCH_API_KEY)
                     for name, value in _WEB_SEARCH_ENDPOINT_DICT.get("headers", {}).items()}
WEB_SEARCH_RESULT_PATH = _WEB_SEARCH_ENDPOINT_DICT.get("results_path", "results")
WEB_SEARCH_RESULT_FIELD = _WEB_SEARCH_ENDPOINT_DICT.get("result_keys", "title,url,content")

# How many results come back. With WEB_SEARCH_BY_THE_MODEL the answer is one of them
WEB_SEARCH_MAX_RESULT = env("WEB_SEARCH_MAX_RESULT", 10, low=1)

# The time limit for one search request (seconds). A wait the endpoint asks for over it
# is not waited: the search is refused instead
WEB_SEARCH_TIMEOUT_SEC = env("WEB_SEARCH_TIMEOUT_SEC", 30, low=1)

# The shortest time between two searches one process makes (seconds). A search that
# comes sooner waits for the rest of it. The request also names this package in its
# User-Agent, and an endpoint that answers 429 or 503 with a Retry-After is left alone
# for as long as it says
WEB_SEARCH_MIN_INTERVAL_SEC = env("WEB_SEARCH_MIN_INTERVAL_SEC", 1.0, low=0)

# The largest reply the search reads (bytes). What is kept from it is bounded by
# WEB_SEARCH_MAX_RESULT and WEB_SEARCH_RESULT_MAX_CHAR; this bounds what an endpoint can
# make this process hold before either of those applies
WEB_SEARCH_MAX_BYTE = env("WEB_SEARCH_MAX_BYTE", 1048576, low=1)

# How much of the results comes back to the model, over the whole returned list. Each
# value of one result is cut to an equal share of it over WEB_SEARCH_MAX_RESULT, and the
# first result is kept even when it alone is over it. 0 means no cap
WEB_SEARCH_RESULT_MAX_CHAR = env("WEB_SEARCH_RESULT_MAX_CHAR", 40000, low=0)

# How long a question may be. It bounds what leaves this machine, and over it the search
# refuses without spending a call
WEB_SEARCH_QUESTION_MAX_CHAR = env("WEB_SEARCH_QUESTION_MAX_CHAR", 300, low=1)

# == Reading one page of the web ==============================
# The tool reads one page of a web_search result by its URL and returns its text. The
# URL goes to a page-reading API this run is allowed to use, which fetches the page
# itself; nothing here fetches a page directly. A URL is refused unless web_search
# returned it since its calls were last counted from 0. The request is made by the
# process that holds the network, as a search is.
#
# The cap on web_fetch calls. Each stage handed the tool counts its calls from 0, except
# that the planning stage and the editing stage after it share one count. 0 hands the
# tool to none and takes it out of the instructions. The tool and its instructions
# appear only when there is a way to read and web_search is handed over as well
# (tools/web.py fetch_available())
WEB_FETCH_MAX_CALL = preset_env("WEB_FETCH_MAX_CALL", low=0)

# The page-reading APIs a run can name. Each entry describes that API's endpoint, with
# {key} standing for WEB_FETCH_API_KEY: "url", "method", "url_param" (the parameter the
# page's URL goes in), "params" (left out where the API needs none), "headers",
# "result_path" (a dotted path to the page in the JSON reply; a list there is read as its
# first element, "" reads the reply itself) and "content_field" (the key of the page
# that holds its text)
WEB_FETCH_PRESET_DICT = {
    "tavily": {"url": "https://api.tavily.com/extract", "method": "POST",
               "url_param": "urls",
               "headers": {"Authorization": "Bearer {key}"},
               "result_path": "results", "content_field": "raw_content"},
    "ollama": {"url": "https://ollama.com/api/web_fetch", "method": "POST",
               "url_param": "url",
               "headers": {"Authorization": "Bearer {key}"},
               "result_path": "", "content_field": "content"},
    "jina": {"url": "https://r.jina.ai/", "method": "POST",
             "url_param": "url",
             "headers": {"Authorization": "Bearer {key}", "Accept": "application/json"},
             "result_path": "data", "content_field": "content"},
}

# Which of WEB_FETCH_PRESET_DICT to read pages through. Empty, there is no page read. It
# is read by whichever process makes the request: a run with a broker sets it, and the
# key, there and not in the container
WEB_FETCH_PROVIDER = choice_env("WEB_FETCH_PROVIDER", "", ("", *WEB_FETCH_PRESET_DICT))
_WEB_FETCH_ENDPOINT_DICT = WEB_FETCH_PRESET_DICT.get(WEB_FETCH_PROVIDER, {})

# The key the preset's API is used with. It goes where the preset puts it and stays
# with the process that makes the request. Unset, and with WEB_FETCH_PROVIDER naming the
# same API as WEB_SEARCH_PROVIDER, it is WEB_SEARCH_API_KEY
WEB_FETCH_API_KEY = env("WEB_FETCH_API_KEY", WEB_SEARCH_API_KEY
                        if WEB_FETCH_PROVIDER == WEB_SEARCH_PROVIDER else "")
if _WEB_FETCH_ENDPOINT_DICT and not WEB_FETCH_API_KEY:
    raise SettingError(f"WEB_FETCH_PROVIDER={WEB_FETCH_PROVIDER} needs its key in "
                       f"WEB_FETCH_API_KEY")


def no_partner_setting_list() -> list[str]:
    """Return the settings that are set while the setting each works with is empty.

    WEB_FETCH_PROVIDER works with WEB_SEARCH_PROVIDER, a key with its provider, and
    SMOKE_FILE and SMOKE_ENV with SMOKE_ARG. The search settings are passed over while
    WEB_SEARCH_MAX_CALL is 0, and the page-reading ones while either that or
    WEB_FETCH_MAX_CALL is 0. It reads the settings as they are when it is called.

    Returns:
        ["<setting> (<the setting it works with> is empty)", ...], empty when every
        setting that is set has the one it works with.
    """
    is_search = WEB_SEARCH_MAX_CALL > 0
    is_fetch = is_search and WEB_FETCH_MAX_CALL > 0
    # (the setting, whether it is set, the setting it works with, that one's value)
    pair_list = [
        ("WEB_FETCH_PROVIDER", is_fetch and WEB_FETCH_PROVIDER,
         "WEB_SEARCH_PROVIDER", WEB_SEARCH_PROVIDER),
        ("WEB_SEARCH_API_KEY", is_search and WEB_SEARCH_API_KEY,
         "WEB_SEARCH_PROVIDER", WEB_SEARCH_PROVIDER),
        ("WEB_FETCH_API_KEY", is_fetch and str_env("WEB_FETCH_API_KEY"),
         "WEB_FETCH_PROVIDER", WEB_FETCH_PROVIDER),
        ("SMOKE_FILE", SMOKE_FILE, "SMOKE_ARG", SMOKE_ARG),
        ("SMOKE_ENV", SMOKE_ENV, "SMOKE_ARG", SMOKE_ARG),
    ]
    return [f"{name} ({partner} is empty)"
            for name, is_set, partner, partner_value in pair_list
            if is_set and not partner_value]


# The endpoint of the named preset, taken from WEB_FETCH_PRESET_DICT and not read from
# the environment. WEB_FETCH_URL is empty when no preset is named
WEB_FETCH_URL = _WEB_FETCH_ENDPOINT_DICT.get("url", "")
WEB_FETCH_METHOD = _WEB_FETCH_ENDPOINT_DICT.get("method", "POST")
WEB_FETCH_URL_PARAM = _WEB_FETCH_ENDPOINT_DICT.get("url_param", "url")
WEB_FETCH_PARAM = _WEB_FETCH_ENDPOINT_DICT.get("params", {})
WEB_FETCH_HEADER = {name: value.format(key=WEB_FETCH_API_KEY)
                    for name, value in _WEB_FETCH_ENDPOINT_DICT.get("headers", {}).items()}
WEB_FETCH_RESULT_PATH = _WEB_FETCH_ENDPOINT_DICT.get("result_path", "data")
WEB_FETCH_CONTENT_FIELD = _WEB_FETCH_ENDPOINT_DICT.get("content_field", "content")

# The time limit for one page request (seconds). A wait the endpoint asks for over it
# is not waited: the read is refused instead
WEB_FETCH_TIMEOUT_SEC = env("WEB_FETCH_TIMEOUT_SEC", 30, low=1)

# The shortest time between two page requests one process makes (seconds), counted
# apart from the searches. An endpoint that answers 429 or 503 with a Retry-After is
# left alone for as long as it says
WEB_FETCH_MIN_INTERVAL_SEC = env("WEB_FETCH_MIN_INTERVAL_SEC", 1.0, low=0)

# The largest reply the read takes in (bytes). A reply over it is not read
WEB_FETCH_MAX_BYTE = env("WEB_FETCH_MAX_BYTE", 4194304, low=1)

# How much of the page's text comes back to the model (characters). The rest is cut
# with the "... (N chars omitted)" marker. 0 means no cap
WEB_FETCH_RESULT_MAX_CHAR = env("WEB_FETCH_RESULT_MAX_CHAR", 40000, low=0)

# == The editing tools ========================================
# The limit on patch-and-test rounds, stated in the instructions and enforced. What is
# counted is not the number of editing calls but the "edit then run_test" rounds; past
# the limit an edit returns ok=False without writing. 0 disables the enforcement and
# leaves the limit stated in the instructions alone
MAX_PATCH_ROUND = preset_env("MAX_PATCH_ROUND", low=0)

# The editing tools run their checks but write nothing. It follows --dry-run and is not
# read from the environment
EDIT_DRY_RUN = False

# The command a container runs over the files its run changed, once the run has ended
# and before its patch is written: a formatter or a code generator, as a shell-quoted
# command line (FORMAT_COMMAND="ruff format"). It is started from the repository root
# with the paths of those files after it and is given TEST_TIMEOUT_SEC. Empty runs none
FORMAT_COMMAND = shell_list_env("FORMAT_COMMAND")

# Whether run_test is handed to the model at all. False takes the tool out of the tool
# list and out of the instructions, and a change run is then never judged tests_red
RUN_TEST = env("RUN_TEST", True)

# The time limit for one test run (seconds)
TEST_TIMEOUT_SEC = env("TEST_TIMEOUT_SEC", 300, low=1)

# How many characters of the tail of the output run_test returns
TEST_OUTPUT_TAIL = env("TEST_OUTPUT_TAIL", 40000, low=0)

# The length of run_test's failed_excerpt (the failure block only), which comes back
# beside "output", the tail of the run's output
FAIL_EXCERPT_CHAR = env("FAIL_EXCERPT_CHAR", 800, low=0)

# How many of the replaced lines replace_line_range echoes in its "stat". The
# instructions name this number
STAT_ECHO_LINE = env("STAT_ECHO_LINE", 3, low=1)

# How much of each echoed line stays. The tool cuts the lines to it; the instructions
# name the line count only
STAT_ECHO_CHAR = env("STAT_ECHO_CHAR", 60, low=1)


# == Running the program ======================================
# What run_project puts in front of the arguments the model passes is built by
# set_primary_ext() below, from RUN_COMMAND or the primary language's run_command_tuple
# (where the repository's top holds one of its run_file_tuple). When both are empty the
# tool is not handed over. The model never supplies argv[0] and cannot start an
# arbitrary binary

# The time limit for one run_project call (seconds)
RUN_TIMEOUT_SEC = env("RUN_TIMEOUT_SEC", 120, low=1)

# How many run_project calls one stage may make. 0 hands the tool to no stage and takes
# it out of the instructions
RUN_PROJECT_MAX_CALL = preset_env("RUN_PROJECT_MAX_CALL", low=0)

# How many characters of the tail of the program's output run_project returns
RUN_OUTPUT_TAIL = env("RUN_OUTPUT_TAIL", 40000, low=0)

# The time limit for the code of one turn (seconds). The default is one test run and
# one program run together. A turn stopped by it comes back as that turn's output and
# the loop goes on. 0 sets no limit
TURN_TIMEOUT_SEC = second_within_max(
    env("TURN_TIMEOUT_SEC", TEST_TIMEOUT_SEC + RUN_TIMEOUT_SEC, low=0),
    "TURN_TIMEOUT_SEC (TEST_TIMEOUT_SEC + RUN_TIMEOUT_SEC when it is not set)")

# How many of the files the run produced are listed
RUN_MAX_FILE = env("RUN_MAX_FILE", 40, low=1)


# == The static check of COBOL ================================
# The dialect ctloom-check-cbl reads COBOL in, as cobc's -std= takes it (ibm, mf, ...).
# Empty is GnuCOBOL's own default. It is asked for at a terminal the first time a
# repository holds COBOL files (knowledge.ask_cobol_dialect), and the answer is written
# to the .env of the directory the command is run from
COBOL_DIALECT = str_env("COBOL_DIALECT")

# The directories of copybooks ctloom-check-cbl searches after the directories of the
# repository that hold a COBOL file, relative to the repository and separated by commas
COBOL_COPYBOOK_DIR = str_env("COBOL_COPYBOOK_DIR")

# The dialects the question offers
COBOL_DIALECT_CHOICE_TUPLE = (("ibm", "IBM Enterprise COBOL"), ("mf", "Micro Focus"))


# == Checking an edit automatically ===========================
# The machine starts the program itself, right after the turn an edit landed on, and
# puts the result at the end of that turn's output.
#
# The arguments the check passes after RUN_COMMAND, e.g.
# "--in {workspace}/sample --out {workspace}/out". Empty turns the check off, which is
# the default
SMOKE_ARG = shell_list_env("SMOKE_ARG")

# The sample the check runs against, as JSON {path relative to the workspace: contents}.
# It is written before every check, into a workspace of the check's own
SMOKE_FILE = json_env("SMOKE_FILE")

# Extra environment variables for the check run, as JSON, e.g. to switch off a feature
# of the target project that would call an LLM
SMOKE_ENV = json_env("SMOKE_ENV")

# How many times the program is started per check, over one workspace. The first start
# empties it and the rest keep what the previous one produced. 1 starts it once
SMOKE_REPEAT = env("SMOKE_REPEAT", 1, low=1)

# How many characters of the failing run's output go into the turn's output, taken from
# the tail run_project returns (RUN_OUTPUT_TAIL)
SMOKE_OUTPUT_TAIL = env("SMOKE_OUTPUT_TAIL", 40000, low=0)

# Whether COMPLETE() is refused once while the last automatic check is failing. As with
# the plan guard it gives way after that one refusal
SMOKE_COMPLETE_GUARD = env("SMOKE_COMPLETE_GUARD", True)


# == The sandbox container ====================================
# What the containers are started with. Empty looks for podman first and then docker,
# and reports which names were tried when neither is on PATH
CONTAINER_ENGINE = env("CONTAINER_ENGINE", "")

# The image the containers run. The default one holds the toolchains of every language
CONTAINER_IMAGE = env("CONTAINER_IMAGE", "codetwine-loom")

# The time limit for the short commands the engine is given besides the run itself
# (seconds): asking what is running, and taking a container away. 0 sets no limit
CONTAINER_COMMAND_TIMEOUT_SEC = 30

# The time limit for building the image (seconds)
IMAGE_BUILD_TIMEOUT_SEC = 3600

# The form the engine's --memory takes, in any case: a number of ASCII digits, which
# may hold a decimal point, then k, m, g, t or p for the unit or nothing for bytes,
# with an "i" and a "b" allowed behind it (2g, 512m, 1.5g, 2gb, 2GiB, 1024)
_MEMORY_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?) ?([kmgtp]?)i?b?",
                        re.IGNORECASE | re.ASCII)

# The units of _MEMORY_RE, each _MEMORY_UNIT_STEP times the one before it, the first
# that many bytes
_MEMORY_UNIT_TEXT = "kmgtp"
_MEMORY_UNIT_STEP = 1024

# The most bytes the engine's --memory holds
_MEMORY_MAX_BYTE = 2 ** 63 - 1


def _memory_byte_count(value: str) -> float | None:
    """Return how many bytes a value of SANDBOX_MEMORY comes to.

    Examples:
        "2g"    -> 2147483648.0
        "1024"  -> 1024.0
        "2 GiB" -> 2147483648.0
        "two"   -> None

    Args:
        value: The setting as it was written.

    Returns:
        The number of bytes, or None for a value that is not of the form _MEMORY_RE.
    """
    match = _MEMORY_RE.fullmatch(value)
    if not match:
        return None
    unit_power = _MEMORY_UNIT_TEXT.index(match[2].lower()) + 1 if match[2] else 0
    return float(match[1]) * float(_MEMORY_UNIT_STEP) ** unit_power


# The memory the container's processes may use, as the engine's --memory takes it
# (_MEMORY_RE), from 1 byte to _MEMORY_MAX_BYTE. What the run writes goes to a directory
# on this machine and is not charged to this
SANDBOX_MEMORY = env("SANDBOX_MEMORY", "2g")
_memory_byte = _memory_byte_count(SANDBOX_MEMORY)
if _memory_byte is None or not 1 <= _memory_byte <= _MEMORY_MAX_BYTE:
    raise SettingError(f"SANDBOX_MEMORY must be a number with k, m, g, t or p behind it "
                       f"for the unit, or nothing for bytes (2g, 512m, 1.5g), that "
                       f"comes to 1 byte or more and to {_MEMORY_MAX_BYTE} bytes or "
                       f"less: {_value_text(SANDBOX_MEMORY)}")

# SANDBOX_MEMORY in bytes
SANDBOX_MEMORY_BYTE = int(_memory_byte)

# How many processes the container may hold at once, threads included
SANDBOX_MAX_PROCESS = env("SANDBOX_MAX_PROCESS", 4096, low=1)

# The container network a run's containers join, by the name the engine knows it under,
# for a repository whose tests need a service another container on that network runs.
# Only a network the engine reports as internal is taken. Empty gives the containers no
# network.
#
# To use it (podman shown; docker takes the same arguments):
#   1. podman network create --internal test-net
#   2. podman run -d --name db --network test-net -e POSTGRES_PASSWORD=test postgres:16
#   3. SANDBOX_NETWORK=test-net
# The tests reach the service under its container's name (db). The variables of this
# machine are not handed to the container: a test that reads where the service is from a
# variable is given it in the test command:
#   TEST_COMMAND="env DATABASE_URL=postgresql://postgres:test@db:5432/postgres pytest -q"
SANDBOX_NETWORK = str_env("SANDBOX_NETWORK")

# How long one agent's container may run (seconds): the limit on everything inside it
# together - RLM_MAX_TURN turns, each of them TURN_TIMEOUT_SEC long and waiting at most
# LLM_BROKER_TIMEOUT_SEC for a reply and TEST_TIMEOUT_SEC / RUN_TIMEOUT_SEC for a
# program - and on an engine that stops answering. A container stopped by it is taken
# away and the run counts as unfinished, which a dispatching run hands out again. 0 sets
# no limit
SANDBOX_TIMEOUT_SEC = env("SANDBOX_TIMEOUT_SEC", 0, low=0)

# Where the toolchain's package cache is kept. Empty puts it under the output directory,
# beside the patch. Filling it needs a network. Agents that each have an output
# directory of their own share one cache when it is set
SANDBOX_CACHE_DIR = str_env("SANDBOX_CACHE_DIR")

# How many times one run may have the package cache filled again. A program started
# after a dependency file was changed has the broker fill the cache first. The broker
# counts only the requests whose dependency files the cache has not been filled for in
# the image, failed fillings included, and a request made any other way is counted the
# same. 0 turns it off
WARM_MAX_REFILL = env("WARM_MAX_REFILL", 3, low=0)

# How long the container that fills the cache may run (seconds). A refill made through
# the broker holds the broker's refill lock while it runs: this is also how long a
# refill can keep the next one waiting. 0 sets no limit
WARM_TIMEOUT_SEC = env("WARM_TIMEOUT_SEC", 0, low=0)

# How long run and sandbox may take to make the overview, the build they start on this
# machine and the container that follows it between them (seconds). An overview stopped
# by it is left as it was, and the run goes on. 0 sets no limit
OVERVIEW_TIMEOUT_SEC = env("OVERVIEW_TIMEOUT_SEC", 0, low=0)

# Whether the build tools must work from the package cache alone: the language's default
# test, run and check commands are given its offline_arg_tuple (languages.py). It is set
# in the containers this tool starts, and is not a setting to set by hand
TOOL_OFFLINE = env("TOOL_OFFLINE", False)


# == What each turn did ========================================
# How many runs' turn logs are kept in <output_dir>/<project name>/turns/. The oldest
# are taken away when a new run starts. 0 writes none
TURN_LOG_KEEP = env("TURN_LOG_KEEP", 20, low=0)

# The least severe line the operator log carries, on stderr and in the run's .log file
# alike: DEBUG, INFO, WARNING, ERROR or CRITICAL. DEBUG also carries every turn, every
# call's token counts and the libraries' own lines
AGENT_LOG_LEVEL = choice_env("AGENT_LOG_LEVEL", "INFO",
                             ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))


# == The values the index supplies ============================
# The language whose test and run commands are used when the index's primary extension
# is not one LANG_DICT covers, or when the index holds neither a definition nor a file of
# a LANG_DICT language.
# set_primary_ext() says in the log that it fell back
FALLBACK_EXT = choice_env("FALLBACK_EXT", "py", tuple(sorted(LANG_DICT)))


def test_command_of(ext: str) -> list[str]:
    """Return the test command for a primary language, as set_primary_ext() sets it.

    Args:
        ext: The primary language's extension. A value not in LANG_DICT takes
            FALLBACK_EXT's.

    Returns:
        TEST_COMMAND split with shlex, or when that is empty the language's default with
        its offline arguments where TOOL_OFFLINE is on. Empty for a language with no
        default and no TEST_COMMAND.
    """
    lang_config = LANG_DICT[ext if ext in LANG_DICT else FALLBACK_EXT]
    return _command_of("TEST_COMMAND", lang_config.test_command_tuple,
                       lang_config.offline_arg_tuple)


def _command_of(setting: str, default_tuple: tuple[str, ...],
                offline_arg_tuple: tuple[str, ...]) -> list[str]:
    """Return a command setting split with shlex, or the language's default for it.

    A setting whose shlex.split comes out empty (TEST_COMMAND=" ", say) takes the
    default. The offline arguments are added to the default alone, and only while
    TOOL_OFFLINE is on: a command written by hand is used as it was written.

    Examples:
        TEST_COMMAND="make test"         -> ["make", "test"]
        unset, ("cargo", "test"), off    -> ["cargo", "test"]
        unset, ("cargo", "test"), on     -> ["cargo", "test", <offline arguments>]
        unset, (), on                    -> []

    Args:
        setting: The environment variable's name, the same as the setting's name. ""
            reads none and returns the default.
        default_tuple: The language's default command, empty for none.
        offline_arg_tuple: The language's offline arguments.

    Returns:
        The command as an argument list, empty when neither the setting nor the
        default names one.
    """
    return (shell_list_env(setting) if setting else []) \
        or list(default_tuple) + list(offline_arg_tuple if TOOL_OFFLINE
                                      and default_tuple else ())


def _has_top_file(name_tuple: tuple[str, ...]) -> bool:
    """Say whether PROJECT_DIR's top directory holds one of some files.

    Args:
        name_tuple: The file names; empty for no condition.

    Returns:
        True for an empty name_tuple, and when one of the names is a file directly in
        PROJECT_DIR.
    """
    return not name_tuple or any(os.path.isfile(os.path.join(PROJECT_DIR, name))
                                 for name in name_tuple)


def _source_count_under(directory: str) -> int:
    """Return how many files under a directory, at any depth, have a LANG_DICT extension."""
    return sum(1 for _, _, file_list in os.walk(directory) for name in file_list
               if os.path.splitext(name)[1][1:].lower() in LANG_DICT)


def test_dir_of(ext: str, root: str) -> str:
    """Return the directory of root a language's tests are taken to be kept in.

    Examples:
        ("py", <only testing/ there>)                          -> "testing"
        ("py", <tests/ with 1 source file, testing/ with 30>)  -> "testing"
        ("py", <none there>)                                   -> "tests"

    Args:
        ext: The language's LANG_DICT key.
        root: Where its tests are run from, as an absolute path.

    Returns:
        Of the language's test_path and test_path_alt_tuple, the directory of root
        holding the most source files (_source_count_under), the earlier one on a tie,
        or test_path when none of them is a directory of root.
    """
    lang_config = LANG_DICT[ext]
    dir_list = [name for name in (lang_config.test_path, *lang_config.test_path_alt_tuple)
                if os.path.isdir(os.path.join(root, name))]
    if len(dir_list) < 2:
        return dir_list[0] if dir_list else lang_config.test_path
    return max(dir_list, key=lambda name: _source_count_under(os.path.join(root, name)))


def test_path_alt_tuple_of(ext: str, test_dir: str) -> tuple[str, ...]:
    """Return the language's directories of tests besides the one its tests are kept in.

    Examples:
        ("ts", "test")    -> ("tests",)
        ("ts", "tests")   -> ("test",)
        ("py", "tests")   -> ("test", "testing")
        ("java", "src/test/java") -> ()

    Args:
        ext: The language's LANG_DICT key.
        test_dir: The directory its tests are kept in (test_dir_of()).

    Returns:
        The language's test_path and test_path_alt_tuple, in order, without test_dir.
    """
    lang_config = LANG_DICT[ext]
    return tuple(name for name in (lang_config.test_path, *lang_config.test_path_alt_tuple)
                 if name != test_dir)


def _test_path_of(value: str) -> str:
    """Return the TEST_PATH setting in the one form a path under it is compared with.

    Examples:
        "tests/"         -> "tests"
        "./tests"        -> "tests"
        "src//test/java" -> "src/test/java"
        "."              -> "."
        ""               -> ""

    Args:
        value: The setting as it was written.

    Returns:
        The path with "." parts, repeated slashes and the trailing slash taken out, or
        "" for an empty value.

    Raises:
        SettingError: When the value is an absolute path or leads out of the repository
            root.
    """
    if not value:
        return ""
    path = posixpath.normpath(value)
    if posixpath.isabs(path) or path == ".." or path.startswith("../"):
        raise SettingError(
            f"TEST_PATH must be a path relative to the repository root, inside it: "
            f"{_value_text(value)}")
    return path


# How the test command is given the test files to run: as paths ("path"), as paths
# after "--" ("separator"), or not at all ("none")
TEST_TARGET_STYLE_TUPLE = ("path", "separator", "none")


def set_primary_ext(ext: str) -> None:
    """Take a language as the primary one and rebuild the test settings.

    PRIMARY_EXT starts as FALLBACK_EXT and is replaced with ext here. TEST_COMMAND,
    TEST_PATH, TEST_PATH_ALT_TUPLE, TEST_TARGET_STYLE, RUN_COMMAND and CHECK_COMMAND are
    rebuilt along with it, and OTHER_CHECK_LIST and OTHER_TEST_LIST are emptied.

    Args:
        ext: The primary language's extension. A value not in LANG_DICT falls back to
            FALLBACK_EXT, with a warning naming the settings to set by hand.

    Raises:
        SettingError: When TEST_TARGET_STYLE is not one of TEST_TARGET_STYLE_TUPLE, or
            TEST_PATH is not a path inside the repository root.
    """
    global PRIMARY_EXT, TEST_COMMAND, TEST_PATH, TEST_PATH_ALT_TUPLE, TEST_TARGET_STYLE
    global RUN_COMMAND, CHECK_COMMAND, OTHER_CHECK_LIST, OTHER_TEST_LIST
    if ext not in LANG_DICT:
        cause = (f"the index's primary extension {ext!r} is not one of the languages "
                 f"with a test command ({', '.join(sorted(LANG_DICT))})") if ext else \
            "the index holds no definition and no file of a language with a test " \
            "command, so it names no primary language"
        logger.warning(
            "%s; falling back to %s. Set TEST_COMMAND / TEST_PATH / TEST_TARGET_STYLE / "
            "RUN_COMMAND / CHECK_COMMAND for this repository", cause, FALLBACK_EXT)
    PRIMARY_EXT = ext if ext in LANG_DICT else FALLBACK_EXT
    lang_config = LANG_DICT[PRIMARY_EXT]
    # Staying empty is allowed for a language with no default (SQL, COBOL, BMS, JCL,
    # CSD): run_test is then not handed over when no other language's tests can be run
    # either, and returns guidance if called with nothing to run
    TEST_COMMAND = test_command_of(PRIMARY_EXT)
    test_path_setting = _test_path_of(str_env("TEST_PATH"))
    TEST_PATH = test_path_setting or test_dir_of(PRIMARY_EXT, PROJECT_DIR)
    # The language's other directories of tests, at the repository root, which hold
    # tests as well while TEST_PATH is not set
    TEST_PATH_ALT_TUPLE = () if test_path_setting \
        else test_path_alt_tuple_of(PRIMARY_EXT, TEST_PATH)
    # Replacing TEST_COMMAND through the environment leaves how targets are passed as
    # the language's; override TEST_TARGET_STYLE itself to change that
    TEST_TARGET_STYLE = choice_of("TEST_TARGET_STYLE", str_env("TEST_TARGET_STYLE"),
                                  TEST_TARGET_STYLE_TUPLE) \
        if str_env("TEST_TARGET_STYLE") else lang_config.test_target
    # Staying empty is allowed: run_project is then not handed over (a call that comes
    # anyway returns guidance instead of running anything)
    RUN_COMMAND = _command_of("RUN_COMMAND", lang_config.run_command_tuple
                              if _has_top_file(lang_config.run_file_tuple) else (), ())
    # The static check the stages that change code run. Staying empty is allowed and
    # runs no check
    CHECK_COMMAND = _command_of("CHECK_COMMAND", lang_config.check_command_tuple,
                                lang_config.offline_arg_tuple)
    # The commands of the other languages are those of one index, and are built again
    # by set_other_ext()
    OTHER_CHECK_LIST = []
    OTHER_TEST_LIST = []


def _other_command_list(ext_list: list[str], setting: str, field: str,
                        primary_ext: str) -> list[tuple[str, list[str]]]:
    """Return the default commands of the languages besides the primary one.

    Args:
        ext_list: The LANG_DICT keys of the languages the index holds a file of,
            sorted. A value not in LANG_DICT is passed over.
        setting: The setting the primary language's command is set by hand under. The
            list is empty while its value in the environment names a command.
        field: The LangConfig attribute holding a language's default command.
        primary_ext: The primary language's LANG_DICT key.

    Returns:
        [(extension, command), ...]: the default of every language among ext_list that
        has one and is not the primary language's, each command once, with the
        language's offline arguments where TOOL_OFFLINE is on and its other_arg_tuple
        after them.
    """
    if shell_list_env(setting):
        return []
    own_tuple_set = {getattr(LANG_DICT[primary_ext], field)}
    command_list = []
    for ext in ext_list:
        lang_config = LANG_DICT.get(ext)
        default_tuple = getattr(lang_config, field) if lang_config is not None else ()
        if not default_tuple or default_tuple in own_tuple_set:
            continue
        own_tuple_set.add(default_tuple)
        command_list.append((ext, _command_of("", default_tuple,
                                              lang_config.offline_arg_tuple)
                             + list(lang_config.other_arg_tuple)))
    return command_list


def set_other_ext(ext_list: Iterable[str]) -> None:
    """Build the static checks and the test commands of the languages besides the primary.

    The languages are those of the index's files. Call it after set_primary_ext().
    OTHER_CHECK_LIST and OTHER_TEST_LIST are rebuilt (_other_command_list). The first is
    left empty while CHECK_COMMAND in the environment names a command and the second
    while TEST_COMMAND does, which is then the one command run.

    Args:
        ext_list: The LANG_DICT keys of the languages the index holds a file of. A
            value not in LANG_DICT is passed over.
    """
    global OTHER_CHECK_LIST, OTHER_TEST_LIST
    sort_list = sorted(set(ext_list))
    OTHER_CHECK_LIST = _other_command_list(sort_list, "CHECK_COMMAND",
                                           "check_command_tuple", PRIMARY_EXT)
    OTHER_TEST_LIST = other_test_list_of(PRIMARY_EXT, sort_list)


def other_test_list_of(primary_ext: str,
                       ext_list: Iterable[str]) -> list[tuple[str, list[str]]]:
    """Return the test commands run besides TEST_COMMAND over an index.

    The index is given by the primary language and the languages of its files.

    Args:
        primary_ext: The primary language (an index's own, or config.PRIMARY_EXT). A
            value not in LANG_DICT takes FALLBACK_EXT's place, as set_primary_ext()
            gives it.
        ext_list: The LANG_DICT keys of the languages the index holds a file of. A
            value not in LANG_DICT is passed over.

    Returns:
        [(extension, command), ...], as _other_command_list() returns them; empty while
        TEST_COMMAND in the environment names a command.
    """
    return _other_command_list(sorted(set(ext_list)), "TEST_COMMAND",
                               "test_command_tuple",
                               primary_ext if primary_ext in LANG_DICT else FALLBACK_EXT)


# The static checks run besides CHECK_COMMAND: (the language's extension, its default
# check command) for every other language the index holds a file of. Empty while
# CHECK_COMMAND in the environment names a command
OTHER_CHECK_LIST: list[tuple[str, list[str]]] = []

# The test commands run besides TEST_COMMAND, in the same form: the default test command
# of every other language the index holds a file of. Empty while TEST_COMMAND in the
# environment names a command
OTHER_TEST_LIST: list[tuple[str, list[str]]] = []

# The primary language: the index's, or one a run takes in its place
# (search.scope.set_primary, set_primary_ext). No setting of config names it
PRIMARY_EXT = FALLBACK_EXT
set_primary_ext(PRIMARY_EXT)  # initial values of what set_primary_ext() rebuilds

# What grep_source walks when glob is omitted: every file under PROJECT_DIR.
# GREP_SKIP_DIR and the copies codetwine wrote are left out
DEFAULT_GLOB_TUPLE = ("**/*",)


# == What a process that is not this one is given =============
# The settings only the process holding the network needs: the credentials, and where
# and how a search is made and a page is read. A process that does not hold the network
# is given none of them
SECRET_NAME_TUPLE = ("LLM_API_KEY", "LLM_API_BASE",
                     "WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY", "WEB_SEARCH_MODEL",
                     "WEB_SEARCH_MAX_TOKEN", "WEB_SEARCH_CONTEXT_SIZE",
                     "WEB_SEARCH_TIMEOUT_SEC", "WEB_SEARCH_MIN_INTERVAL_SEC",
                     "WEB_FETCH_PROVIDER", "WEB_FETCH_API_KEY",
                     "WEB_FETCH_TIMEOUT_SEC", "WEB_FETCH_MIN_INTERVAL_SEC")


def proxy_name_list() -> list[str]:
    """Return the proxy variables this process holds a value for.

    Examples:
        HTTPS_PROXY=http://proxy.corp.example:3128, no_proxy=localhost
            -> ["HTTPS_PROXY", "no_proxy"]

    Returns:
        The names of PROXY_NAME_TUPLE and the same names in lower case that are set
        and not empty, each as the environment holds it.
    """
    return [name for upper_name in PROXY_NAME_TUPLE
            for name in (upper_name, upper_name.lower()) if os.environ.get(name)]


class _TunnelConnection(http.client.HTTPConnection):
    """A connection that reads host and port as urllib's https proxy connection does.

    A port that is absent or empty is 443.
    """

    default_port = http.client.HTTPS_PORT


def proxy_with_port(proxy: str) -> str:
    """Write an http:// proxy's URL with its scheme and its port.

    Examples:
        "http://proxy.corp.example"            -> "http://proxy.corp.example:80"
        "http://proxy.corp.example:"           -> "http://proxy.corp.example:80"
        "me:pw@proxy.corp.example/"            -> "http://me:pw@proxy.corp.example:80/"
        "proxy.corp.example:3128"              -> "http://proxy.corp.example:3128"
        "http://[fd00::1]"                     -> "http://[fd00::1]:80"
        "http://proxy.corp.example:3128"       -> "http://proxy.corp.example:3128"
        "https://proxy.corp.example"           -> "https://proxy.corp.example"

    Args:
        proxy: A proxy's URL as a proxy variable holds it. One without "://" is read
            as http://.

    Returns:
        For a scheme of http: proxy with "http://" before it when it has no scheme,
        and ":80" after its host in place of a port that is absent or empty.
        Otherwise proxy as it is.
    """
    scheme, _, rest = proxy.partition("://") if "://" in proxy else ("http", "", proxy)
    if scheme.lower() != "http":
        return proxy
    user_part, at, host_part = rest.rpartition("@")
    host_port, slash, after = host_part.partition("/")
    host_port = host_port.removesuffix(":")
    has_port = ("]:" in host_port if host_port.startswith("[")
                else ":" in host_port)
    if host_port and not has_port:
        host_port += ":80"
    head = proxy[:len(proxy) - len(rest)] or "http://"
    return f"{head}{user_part}{at}{host_port}{slash}{after}"


def _host_key(host: str) -> str:
    """Write a host a connection is made to one way, for two of them to be compared.

    Args:
        host: The host, as an HTTP client resolves it.

    Returns:
        host in lower case, or the address it names when it is an IP address.
    """
    host = host.lower()
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host


def _httpx_proxy_reading(url: str) -> tuple[str, int | None, bytes | None]:
    """Read a proxy's URL as httpx does.

    Args:
        url: The proxy's URL, with its scheme.

    Returns:
        The host it connects to, the port, and the user and the password it sends
        joined by ":" (None when it sends none).
    """
    proxy = httpx.Proxy(url)
    port = proxy.url.port
    if port is None:
        port = 443 if proxy.url.scheme == "https" else 80
    auth = proxy.raw_auth
    return (proxy.url.raw_host.decode("ascii"), port,
            None if auth is None else b":".join(auth))


class _NoLookupResolver:
    """A resolver that looks nothing up."""

    async def resolve(self, host: str, port: int = 0, family: int = 0) -> list:
        """Return no address."""
        return []


def _aiohttp_connect_check(host: str, port: int) -> None:
    """Make the checks aiohttp makes of a host before it connects to it, with no lookup.

    aiohttp's own TCPConnector._resolve_host() is run over a stand-in connector whose
    resolver looks nothing up, as a coroutine taken one step.

    Args:
        host: The host, as aiohttp's connector is given it.
        port: The port.

    Raises:
        Exception: What aiohttp raises for the host, such as InvalidUrlClientError for
            an IPv4 address that is not written as four numbers. RuntimeError when the
            check does not end in one step.
    """
    from aiohttp.connector import TCPConnector

    connector = types.SimpleNamespace(_family=0, _use_dns_cache=False, _closed=False,
                                      _resolver=_NoLookupResolver())
    coroutine = TCPConnector._resolve_host(connector, host, port)
    try:
        coroutine.send(None)
    except StopIteration:
        return
    coroutine.close()
    raise RuntimeError("aiohttp's check of the host did not end in one step")


def _aiohttp_proxy_reading(url: str) -> tuple[str, int | None, bytes | None]:
    """Read a proxy's URL as aiohttp does.

    The checks aiohttp makes of the host before connecting are included
    (_aiohttp_connect_check).

    Args:
        url: The proxy's URL, with its scheme.

    Returns:
        The host it connects to, the port, and the user and the password it sends
        joined by ":" (None when it sends none).
    """
    import yarl
    from aiohttp.helpers import encode_basic_auth

    proxy = yarl.URL(url)
    auth = None
    if proxy.raw_user or proxy.raw_password:
        header = encode_basic_auth(proxy.user or "", proxy.password or "", "latin1")
        auth = base64.b64decode(header.partition(" ")[2])
    host = proxy.raw_host or ""
    if host:
        # The connector is given the host with a run of dots at its end as one dot
        _aiohttp_connect_check(
            host.rstrip(".") + "." if host.endswith("..") else host, proxy.port)
    return host, proxy.port, auth


def _urllib_proxy_reading_list(url: str) -> list[tuple[str, int | None, bytes | None]]:
    """Read an http:// proxy's URL as urllib does.

    The URL is the one proxy_with_port() gives, read as ProxyHandler does, and its host
    and port as http.client reads them for an http request and for an https one.

    Args:
        url: The proxy's URL, with its scheme.

    Returns:
        For an http request and for an https one: the host it connects to (a host
        that is not ASCII encoded as the socket module does), the port, and the user
        and the password it sends joined by ":" (None when it sends none).
    """
    _, user, password, host_port = urllib.request._parse_proxy(proxy_with_port(url))
    auth = (f"{urllib.parse.unquote(user)}:{urllib.parse.unquote(password)}".encode()
            if user and password else None)
    reading_list = []
    for connection_class in (http.client.HTTPConnection, _TunnelConnection):
        connection = connection_class(urllib.parse.unquote(host_port))
        host = connection.host
        if not host.isascii():
            host = host.encode("idna").decode("ascii")
        reading_list.append((host, connection.port, auth))
    return reading_list


def _is_port_written(url: str) -> bool:
    """Say whether a proxy's URL writes a port.

    Args:
        url: The proxy's URL, with its scheme.

    Returns:
        True when yarl reads a port written in it.
    """
    import yarl

    return yarl.URL(url).explicit_port is not None


def _proxy_reading_list(scheme: str, url: str) -> list[tuple[str, int | None, bytes | None]]:
    """Read a proxy's URL as httpx, aiohttp and, for an http:// proxy, urllib do.

    Args:
        scheme: The URL's scheme, in lower case.
        url: The proxy's URL, with its scheme.

    Returns:
        The readings of httpx and aiohttp, and for an http:// proxy those of urllib
        (_urllib_proxy_reading_list), each as (host, port, user and password).

    Raises:
        Exception: Whatever a client's reading raises.
    """
    reading_list = [_httpx_proxy_reading(url), _aiohttp_proxy_reading(url)]
    if scheme == "http":
        reading_list += _urllib_proxy_reading_list(url)
    return reading_list


def _split_host_port(host_port: str) -> tuple[str, str]:
    """Split the host and the port of a proxy's URL.

    Examples:
        "proxy.corp.example:3128" -> ("proxy.corp.example", "3128")
        "[fd00::1]:3128"          -> ("[fd00::1]", "3128")
        "proxy.corp.example"      -> ("proxy.corp.example", "")

    Args:
        host_port: What stands between the user and password, and the path.

    Returns:
        The host, and the port as it is written ("" when none is).
    """
    if host_port.startswith("[") and "]" in host_port:
        end = host_port.index("]") + 1
        return host_port[:end], host_port[end:].removeprefix(":")
    host, colon, port = host_port.rpartition(":")
    return (host, port) if colon else (host_port, "")


# What proxy_url_error() says of a port no HTTP client of this tool connects to
_PROXY_PORT_TEXT = "names a port that is not a number from 1 to 65535"


def _credential_text(user_part: str) -> str:
    """Say how to write a proxy's user and password that a client cannot read.

    Args:
        user_part: What stands before the last "@" of the URL, after the scheme.

    Returns:
        For a user or a password that holds a character outside ASCII once %xx is read,
        that no way of writing it is sent alike. Otherwise to write it with %xx.
    """
    if not urllib.parse.unquote(user_part).isascii():
        return ("holds a user or a password with a character outside ASCII, which the "
                "HTTP clients of this tool send differently, written with %xx or not: "
                "use a user and a password in ASCII")
    return ("holds a user or a password not every HTTP client of this tool can read: "
            "write each character of them other than an ASCII letter or digit as %xx")


def _credential_mismatch_text(user_part: str) -> str:
    """Say why the HTTP clients of this tool send different users and passwords.

    Args:
        user_part: What stands before the last "@" of the URL, after the scheme.

    Returns:
        _credential_text() for a user or a password outside ASCII; for one of the two
        that is empty, to write both; otherwise that they are not sent alike.
    """
    if not urllib.parse.unquote(user_part).isascii():
        return _credential_text(user_part)
    user, colon, password = user_part.partition(":")
    if not (user and colon and password):
        return ("names a user without a password or a password without a user, which "
                "not every HTTP client of this tool sends: write both")
    return ("is not read as the same user and password by every HTTP client of this "
            "tool: write each character of them other than an ASCII letter or digit "
            "as %xx")


def _host_mismatch_text(host: str) -> str:
    """Say why the HTTP clients of this tool connect to different hosts or ports.

    Args:
        host: The host as the URL writes it.

    Returns:
        The sentence for a host written with "%", for a host outside ASCII, and for
        any other.
    """
    if "%" in host:
        return ("names a host written with '%', which the HTTP clients of this tool "
                "read differently: write it without %xx (an IPv6 address with a zone "
                "cannot be used)")
    if not host.isascii():
        return ("names a host the HTTP clients of this tool encode differently: write "
                "it in ASCII, a name outside ASCII in its xn-- form")
    return ("is not read as the same host and port by every HTTP client of this tool: "
            "write it as http://host:port")


def _unread_part_text(scheme: str, user_part: str, host_port: str) -> str:
    """Say which part of a proxy's URL an HTTP client of this tool cannot read.

    The URL is read again by every client without its user and password: when that is
    read, the user or the password is the part (_credential_text). Otherwise it is read
    with its host replaced by _STAND_IN_PROXY_HOST: when that is not read either, the
    port is the part, and otherwise the host. When _STAND_IN_PROXY_HOST alone is not
    read, no part is named.

    Args:
        scheme: The URL's scheme, in lower case.
        user_part: What stands before the last "@", after the scheme ("" when none).
        host_port: What stands between that and the path.

    Returns:
        Why the URL is not read, in one sentence that holds nothing of it.
    """
    def is_read(rest: str) -> bool:
        """Say whether every client reads the URL with rest after its scheme."""
        try:
            _proxy_reading_list(scheme, f"{scheme}://{rest}")
        except Exception:
            return False
        return True

    port = _split_host_port(host_port)[1]
    if not is_read(_STAND_IN_PROXY_HOST) or (is_read(host_port) and not user_part):
        return ("cannot be read by every HTTP client of this tool: write it as "
                "http://host:port")
    if is_read(host_port):
        return _credential_text(user_part)
    if port and not is_read(f"{_STAND_IN_PROXY_HOST}:{port}"):
        return _PROXY_PORT_TEXT
    return ("names a host not every HTTP client of this tool can connect to: write it "
            "in ASCII, a name outside ASCII in its xn-- form and an IPv4 address as "
            "four numbers from 0 to 255")


def proxy_url_error(value: str, port_needed: bool = False) -> str:
    """Say what keeps a proxy's URL from being read the same by the HTTP clients.

    The clients are httpx, aiohttp and, for an http:// proxy, urllib
    (_proxy_reading_list). A value without "://" is read as an http:// URL. A value is
    read when it holds no blank or control character, its scheme is one of
    PROXY_SCHEME_TUPLE, no "#", "/" or "?" stands in its user or password (they are
    written %23, %2F and %3F), nothing but one "/" follows the host and the port, and
    every client reads it without an error into the same host to connect to, the same
    port, from 1 to 65535, and the same user and password to send.

    Examples:
        "http://proxy.corp.example:3128"              -> ""
        "http://user:p%23ss@proxy.corp.example:3128"  -> ""
        "proxy.corp.example:3128"                     -> ""
        "http://proxy.bücher.example:3128"            -> ""
        "socks5://proxy.corp.example:1080"            -> "points at a SOCKS proxy, ..."
        "http://user:pa#ss@proxy.corp.example:3128"   -> "holds a '#', '/' or '?' ..."
        "http://straße.example:3128"                  -> "names a host the HTTP ..."
        "http://proxy.corp.example:31a8"              -> "names a port that is ..."
        "http://user:pässwort@proxy.corp.example:3128"
            -> "holds a user or a password with a character outside ASCII, ..."

    Args:
        value: The URL, as a proxy variable holds it.
        port_needed: Whether a value that writes no port is refused.

    Returns:
        "" when it is read. Otherwise why it is not, in one sentence that holds
        nothing of the value.
    """
    if any(char.isspace() or not char.isprintable() for char in value):
        return "holds a blank or a control character"
    scheme, _, rest = value.partition("://") if "://" in value else ("http", "", value)
    scheme = scheme.lower()
    if scheme.startswith("socks"):
        return ("points at a SOCKS proxy, which this tool cannot go through: unset it, "
                "or point it at an http:// or https:// proxy")
    if scheme not in PROXY_SCHEME_TUPLE:
        return ("is not an http:// or https:// URL, the only proxies this tool goes "
                "through")
    user_part, _, host_part = rest.rpartition("@")
    if any(char in user_part for char in "#/?"):
        return ("holds a '#', '/' or '?' in the user or the password: write them as "
                "%23, %2F and %3F")
    host_port, _, after = host_part.partition("/")
    if after or "#" in host_port or "?" in host_port:
        return "holds a path, a query or a fragment: write it as http://host:port"
    if not host_port or host_port.startswith(":"):
        return "names no host"
    url = f"{scheme}://{rest}"
    # An error of a client's reading is answered by the part of the URL it stands in,
    # its text left out
    try:
        reading_list = _proxy_reading_list(scheme, url)
        is_port_written = _is_port_written(url)
    except Exception:
        return _unread_part_text(scheme, user_part, host_port)
    host_set = {_host_key(host) for host, _, _ in reading_list}
    port_set = {port for _, port, _ in reading_list}
    auth_set = {auth for _, _, auth in reading_list}
    if "" in host_set:
        return "names no host"
    if any(not isinstance(port, int) or not 1 <= port <= 65535 for port in port_set):
        return _PROXY_PORT_TEXT
    if len(host_set) > 1 or len(port_set) > 1:
        return _host_mismatch_text(_split_host_port(host_port)[0])
    if len(auth_set) > 1:
        return _credential_mismatch_text(user_part)
    if port_needed and not is_port_written:
        return "names no port: write it as http://host:port or host:port"
    return ""


def _proxy_source_name(key: str, value: str) -> str:
    """Name the variable urllib.request.getproxies() took one proxy from.

    Args:
        key: The key it gave the proxy under: "https", "http", "all" or "no".
        value: The proxy it gave.

    Returns:
        The first of <key>_proxy, <KEY>_PROXY and the other spellings of that name
        that holds value, as the environment holds it.
    """
    lower_name = f"{key}_proxy"
    return next((name for name in (lower_name, lower_name.upper(), *os.environ)
                 if name.lower() == lower_name and os.environ.get(name) == value),
                lower_name.upper())


def proxy_error_text() -> str:
    """Name the first proxy variable read whose value proxy_url_error() refuses.

    When none is refused, the NO_PROXY no_proxy_error() refuses for LLM_API_BASE is
    named. The values looked at are those urllib.request.getproxies() gives for https,
    http, all and no: a variable whose name in lower case holds a value is read in place
    of the same name in upper case.

    Examples:
        ALL_PROXY=socks5://proxy.corp.example:1080
            -> "ALL_PROXY points at a SOCKS proxy, ..."
        HTTPS_PROXY=socks5://proxy.corp.example:1080,
        https_proxy=http://proxy.corp.example:3128 -> ""

    Returns:
        "<name> <why>", the value left out. "" when each of them is read.
    """
    proxy_dict = urllib.request.getproxies()
    for key in _PROXY_KEY_TUPLE:
        value = proxy_dict.get(key, "")
        if not value:
            continue
        error = proxy_url_error(value)
        if error:
            return f"{_proxy_source_name(key, value)} {error}"
    no_proxy = proxy_dict.get("no", "")
    error = no_proxy_error(no_proxy, LLM_API_BASE) if no_proxy else ""
    return f"{_proxy_source_name('no', no_proxy)} {error}" if error else ""


def _model_endpoint(model: str) -> str:
    """Return the endpoint litellm calls a model at when no api_base is passed.

    litellm's own get_llm_provider() gives it for the providers it resolves an endpoint
    for, and the provider's model information (get_api_base()) for the others it has.
    litellm is not asked about a model of a provider it signs in to as it resolves it
    (providercall.look_up_model): a warning naming the model, once for each model, says
    that NO_PROXY is not checked against its endpoint.

    Args:
        model: The model, in litellm form.

    Returns:
        The URL, from the provider's variable or the provider's default. "" when
        litellm gives none with "://", whatever it raises, and for a model it is not
        asked about.
    """
    from codetwine_loom import providercall
    import litellm
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    try:
        found = providercall.look_up_model(litellm.get_llm_provider, model, None)
        if found is None:
            if model not in _UNASKED_ENDPOINT_MODEL_SET:
                _UNASKED_ENDPOINT_MODEL_SET.add(model)
                logger.warning("the endpoint of %s is not known without signing in to "
                               "its provider, so NO_PROXY is not checked against it",
                               model)
            return ""
        _, provider, _, endpoint = found
        if not endpoint:
            model_info = ProviderConfigManager.get_provider_model_info(
                model=model, provider=LlmProviders(provider))
            endpoint = model_info.get_api_base() if model_info is not None else ""
    except Exception:
        return ""
    return endpoint if isinstance(endpoint, str) and "://" in endpoint else ""


def model_endpoint_dict(api_base: str) -> dict[str, str]:
    """Return the endpoints the model calls of this process go to, by the name of each.

    Examples:
        "http://llm.corp.example:8080/v1"
            -> {"LLM_API_BASE": "http://llm.corp.example:8080/v1"}
        "" with OPENAI_BASE_URL=http://llm.corp.example:8080/v1 and the models
        openai/m and llamafile/m
            -> {"OPENAI_BASE_URL": "http://llm.corp.example:8080/v1",
                "the default endpoint of llamafile/m": "http://127.0.0.1:8080/v1"}

    Args:
        api_base: LLM_API_BASE, or the address that stands in its place.

    Returns:
        {"LLM_API_BASE": api_base} when api_base is not empty. Otherwise each variable
        of the environment whose name _ENDPOINT_NAME_RE matches whole and whose value
        starts with a scheme of _ENDPOINT_SCHEME_TUPLE, whatever model reads it, then
        the endpoint litellm calls each model of the settings at (_model_endpoint)
        under the name of a variable holding the same URL or as the model's default
        endpoint, and _PROVIDER_ENDPOINT_URL standing for the providers' own endpoints
        when that of a model is not known.
    """
    if api_base:
        return {"LLM_API_BASE": api_base}
    endpoint_dict = {name: value for name, value in sorted(os.environ.items())
                     if _ENDPOINT_NAME_RE.fullmatch(name)
                     and value.lower().startswith(_ENDPOINT_SCHEME_TUPLE)}
    for model in dict.fromkeys(setting_value_dict("_MODEL").values()):
        if not model:
            continue
        endpoint = _model_endpoint(model)
        if not endpoint:
            endpoint_dict[_PROVIDER_ENDPOINT_NAME] = _PROVIDER_ENDPOINT_URL
        elif endpoint not in endpoint_dict.values():
            name = next((name for name, value in os.environ.items() if value == endpoint),
                        f"the default endpoint of {model}")
            endpoint_dict[name] = endpoint
    return endpoint_dict


def with_global(function: Callable, name: str, value: object) -> Callable:
    """Return a copy of a function whose code reads value under one global name.

    The copy runs the function's own code, with its name, its positional defaults and
    its closure, over a copy of its module's globals in which name is value.
    Keyword-only defaults are not carried over. The function and its module are left as
    they are.

    Args:
        function: A plain Python function.
        name: The global name the function's code reads.
        value: What the copy's code reads under name.

    Returns:
        The new function.
    """
    return types.FunctionType(function.__code__, {**function.__globals__, name: value},
                              function.__name__, function.__defaults__,
                              function.__closure__)


def _httpx_no_proxy_pattern_list(proxy_dict: Mapping[str, str]) -> list[tuple[object, bool]]:
    """Read the proxy variables as httpx does.

    httpx's own get_environment_proxies() is run with proxy_dict in place of what
    urllib.request.getproxies() gives, and its keys are ordered as httpx.Client orders
    its mounts.

    Args:
        proxy_dict: The proxy variables, as urllib.request.getproxies() gives them.

    Returns:
        (pattern, whether a request it matches goes through a proxy) for each key, in
        the order a request is matched against them.

    Raises:
        Exception: Whatever httpx raises for a key it cannot read, which keeps
            httpx.Client from being made.
    """
    from httpx import _utils as httpx_utils

    read_proxy = httpx_utils.get_environment_proxies
    mount_dict = types.FunctionType(
        read_proxy.__code__,
        {**read_proxy.__globals__, "getproxies": lambda: dict(proxy_dict)})()
    return [(pattern, mount_dict[pattern.pattern] is not None)
            for pattern in sorted(httpx_utils.URLPattern(key) for key in mount_dict)]


def _httpx_goes_through_proxy(pattern_list: list[tuple[object, bool]],
                              url: httpx.URL) -> bool:
    """Say whether httpx sends a request through a proxy.

    Args:
        pattern_list: What _httpx_no_proxy_pattern_list() returned.
        url: The request's URL.

    Returns:
        The answer of the first pattern that matches url; False when none does.
    """
    return next((is_proxy for pattern, is_proxy in pattern_list
                 if pattern.matches(url)), False)


def _aiohttp_goes_through_proxy(proxy_dict: Mapping[str, str], url: httpx.URL) -> bool:
    """Say whether litellm's aiohttp transport sends a request through a proxy.

    litellm's own LiteLLMAiohttpTransport._proxy_from_env() is run with
    urllib.request's getproxies() and proxy_bypass() reading proxy_dict in place of the
    environment. An error it raises sends the request directly, as litellm's
    _get_proxy_settings() does.

    Args:
        proxy_dict: The proxy variables, as urllib.request.getproxies() gives them.
        url: The request's URL.

    Returns:
        True when the request goes through a proxy.
    """
    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

    read_proxy = LiteLLMAiohttpTransport._proxy_from_env
    request_module = types.SimpleNamespace(
        getproxies=lambda: dict(proxy_dict),
        proxy_bypass=lambda host: urllib.request.proxy_bypass_environment(
            host, dict(proxy_dict)))
    read_proxy_here = with_global(read_proxy, "urllib",
                                  types.SimpleNamespace(request=request_module))
    try:
        return read_proxy_here(types.SimpleNamespace(proxy_cache={}), url) is not None
    except Exception:
        return False


def _no_proxy_entry_kind(entry: str) -> str:
    """Name the kind of a NO_PROXY entry, a key of _NO_PROXY_ENTRY_TEXT_DICT.

    Args:
        entry: The entry, without the blanks around it.

    Returns:
        "scheme", "port", "dot" or "other".
    """
    if "://" in entry:
        return "scheme"
    if entry.startswith("["):
        has_port = "]:" in entry
    else:
        try:
            ipaddress.ip_address(entry)
            has_port = False
        except ValueError:
            has_port = ":" in entry
    if has_port:
        return "port"
    return "dot" if entry.startswith(".") else "other"


def _no_proxy_entry_text(proxy_dict: Mapping[str, str], url: httpx.URL) -> str:
    """Name the kind of NO_PROXY entry httpx and aiohttp read differently for a request.

    How to write it instead is named with it. Each entry is read alone, by both clients.
    When none is read differently alone, the entries are read differently together: "*"
    among them, or another kind.

    Args:
        proxy_dict: The proxy variables, as urllib.request.getproxies() gives them.
        url: The request's URL.

    Returns:
        "<kind>: <how to write it>", from _NO_PROXY_ENTRY_TEXT_DICT.
    """
    entry_list = [entry.strip() for entry in proxy_dict["no"].split(",") if entry.strip()]
    kind = "star" if "*" in entry_list else "other"
    for entry in entry_list:
        entry_dict = {**proxy_dict, "no": entry}
        if (_httpx_goes_through_proxy(_httpx_no_proxy_pattern_list(entry_dict), url)
                != _aiohttp_goes_through_proxy(entry_dict, url)):
            kind = _no_proxy_entry_kind(entry)
            break
    entry_text, write_text = _NO_PROXY_ENTRY_TEXT_DICT[kind]
    return f"{entry_text}: {write_text}"


def no_proxy_error(no_proxy: str, api_base: str, https_proxy: str = "") -> str:
    """Say what keeps two clients from reading the hosts reached without a proxy alike.

    The clients are httpx and aiohttp. Each client's own reading
    (_httpx_no_proxy_pattern_list, _aiohttp_goes_through_proxy) is asked whether a
    request to each endpoint of model_endpoint_dict() goes through the proxy. The value
    is refused when httpx cannot read an entry, with a proxy set or not, and when the
    two answer differently for an endpoint.

    Examples:
        "localhost,llm.corp.example", "http://llm.corp.example:8080/v1" -> ""
        "llm.corp.example:8080", "http://llm.corp.example:8080/v1"
            -> "is not read alike ... an entry with a port (host:port): ..."
        "localhost,*", "" -> "is not read alike ... '*' among other entries: ..."

    Args:
        no_proxy: The value of NO_PROXY.
        api_base: LLM_API_BASE, or the address that stands in its place.
        https_proxy: The URL of HTTPS_PROXY to read in place of the one the
            environment gives; "" keeps that one.

    Returns:
        "" when they read it alike, or when httpx reads it and no proxy is set.
        Otherwise why not, in one sentence that holds nothing of the value.
    """
    proxy_dict = urllib.request.getproxies()
    if https_proxy:
        proxy_dict["https"] = https_proxy
    proxy_dict["no"] = no_proxy
    try:
        pattern_list = _httpx_no_proxy_pattern_list(proxy_dict)
    except Exception:
        return ("holds an entry not every HTTP client of this tool can read: write each "
                "host in ASCII, and an IPv6 address without '[' and ']'")
    if not any(proxy_dict.get(key) for key in _PROXY_KEY_TUPLE):
        return ""
    for name, endpoint in model_endpoint_dict(api_base).items():
        try:
            url = httpx.URL(endpoint)
        except Exception:
            continue
        # A reading that cannot be made at all refuses the value, its error left out
        try:
            if (_httpx_goes_through_proxy(pattern_list, url)
                    == _aiohttp_goes_through_proxy(proxy_dict, url)):
                continue
            entry_text = _no_proxy_entry_text(proxy_dict, url)
        except Exception:
            return ("cannot be read by every HTTP client of this tool: write each host "
                    "alone, in ASCII, separated by commas")
        return (f"is not read alike by every HTTP client of this tool for {name}, so "
                f"that some calls would go through the proxy and some not, over "
                f"{entry_text}")
    return ""


def credential_dict() -> dict[str, str]:
    """Return the keyword arguments that put this process's credentials on a model call.

    Returns:
        {"api_key": LLM_API_KEY, "api_base": LLM_API_BASE}, each only when it is set.
    """
    return {name: value
            for name, value in (("api_key", LLM_API_KEY), ("api_base", LLM_API_BASE))
            if value}


# How the output of a program this process starts is read: as UTF-8 text, a byte that
# is not UTF-8 read as U+FFFD. Passed to subprocess.run() as keyword arguments
CHILD_TEXT_DICT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def shell_env() -> dict[str, str]:
    """Return the environment of the shell this process was started from.

    It is this process's own environment with each name of _OPTION_FILL_DICT that still
    holds what the command line gave it put back as it was before, then each name of
    _OWN_FILL_DICT that still holds what this module wrote, or is still taken out, put
    back as it was before, then each name of _ENV_FILE_FILL_DICT that holds what the
    .env file gave it put back as the shell held it: unset, or with the value held
    before.

    Returns:
        The environment, as a new dict.
    """
    env_dict = dict(os.environ)
    for fill_dict in (_OPTION_FILL_DICT, _OWN_FILL_DICT, _ENV_FILE_FILL_DICT):
        for name, (prior_value, fill_value) in fill_dict.items():
            if env_dict.get(name) == fill_value:
                if prior_value is None:
                    env_dict.pop(name, None)
                else:
                    env_dict[name] = prior_value
    return env_dict


def child_env(extra_dict: dict[str, str] | None = None,
              base_dict: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment a program this process starts is given.

    It is this process's own environment (base_dict in its place when one is given)
    without SECRET_NAME_TUPLE, without the providers' own variables the models read
    (llm.provider_variable_tuple), without LLM_BROKER and without GIT_PLACE_NAME_TUPLE
    and GIT_ANSWER_NAME_TUPLE. Everything else is kept: what the toolchain reads (PATH,
    HOME, the cache directories) reaches the program unchanged.

    Args:
        extra_dict: Values to add on top, which win over what is kept. None adds nothing.
        base_dict: The environment to start from in place of this process's own.
            None starts from this process's own.

    Returns:
        The environment, as a new dict.
    """
    from codetwine_loom import llm
    drop_set = (set(SECRET_NAME_TUPLE) | set(llm.provider_variable_tuple())
                | {"LLM_BROKER"} | set(GIT_PLACE_NAME_TUPLE) | set(GIT_ANSWER_NAME_TUPLE))
    base_dict = os.environ if base_dict is None else base_dict
    return {**{name: value for name, value in base_dict.items() if name not in drop_set},
            **(extra_dict or {})}


def own_child_env(extra_dict: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment a process of this tool given no credential is started with.

    It is child_env() with NO_ENV_FILE_ENV set to NO_ENV_FILE_VALUE.

    Args:
        extra_dict: Values to add on top, which win over what is kept. None adds nothing.

    Returns:
        The environment, as a new dict.
    """
    return child_env({NO_ENV_FILE_ENV: NO_ENV_FILE_VALUE, **(extra_dict or {})})


# What a process of this tool runs first. It loads the package its first argument names
# from the file its second argument names, and runs the command line with the rest
_OWN_START = (
    "import importlib.util, os, sys\n"
    "name, init_path = sys.argv.pop(1), sys.argv.pop(1)\n"
    "spec = importlib.util.spec_from_file_location(\n"
    "    name, init_path, submodule_search_locations=[os.path.dirname(init_path)])\n"
    "sys.modules[spec.name] = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(sys.modules[spec.name])\n"
    "importlib.import_module(spec.name + '.__main__').main()\n")


def own_argv(argument_list: list[str]) -> list[str]:
    """Return the command line that starts this tool in a process of its own.

    The process runs the interpreter of this one with -P (its working directory is no
    place it imports modules from), and loads this package from the file this process
    loaded it from, wherever else the name is found.

    Args:
        argument_list: The subcommand and its arguments.

    Returns:
        The argument list, the interpreter first.
    """
    package = sys.modules[__name__.partition(".")[0]]
    return [sys.executable, "-P", "-c", _OWN_START, package.__name__, package.__file__,
            *argument_list]


# The settings a language's own check under sandbox/ reads from its environment. They
# reach the target repository's commands
PROJECT_SETTING_TUPLE = ("COBOL_DIALECT", "COBOL_COPYBOOK_DIR")


def project_env(extra_dict: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment a command of the target repository is given.

    It is child_env() without the settings this module reads, except
    PROJECT_SETTING_TUPLE.

    Args:
        extra_dict: Values to add on top, which win over what is kept. None adds nothing.

    Returns:
        The environment, as a new dict.
    """
    drop_set = set(env_name_tuple()) - set(PROJECT_SETTING_TUPLE)
    return {**{name: value for name, value in child_env().items() if name not in drop_set},
            **(extra_dict or {})}


# == The design docs ==========================================
# The model codetwine writes the design docs with (litellm format). codetwine's own
# LLM_MODEL is not read. Left unset it follows MEDIUM_MODEL
DOC_CREATE_MODEL = env("DOC_CREATE_MODEL", MEDIUM_MODEL)

# The effort codetwine writes the design documents at: "off" turns thinking off, and
# "low" / "medium" / "high" ask for adaptive thinking at that effort. A model that
# cannot stop thinking (litellm's thinking_always_on) is asked for "low" where "off" is
# set. codetwine's own DOC_EFFORT is not read
DOC_CREATE_EFFORT = choice_env("DOC_CREATE_EFFORT", "low", DOC_EFFORT_TUPLE)
