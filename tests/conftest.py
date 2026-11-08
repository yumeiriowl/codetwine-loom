"""Synthetic data (Java, JavaScript), for defects the real data cannot expose.

Python data has no same-name collisions, so step 1's definition-key splitting, step 3's
name lookup and search step 2's is_constant cannot be exercised against it.
"""

import collections
import errno
import importlib
import os
import signal
import socket
import subprocess
import threading

import dotenv
import pytest

_LOCAL_HOST_TUPLE = ("127.0.0.1", "::1", "localhost", "", "0.0.0.0")
_socket_connect = socket.socket.connect
_socket_connect_ex = socket.socket.connect_ex
_socket_getaddrinfo = socket.getaddrinfo


def _refuse_other_machine(address: object) -> None:
    """Raise for an address that is not on this machine.

    Args:
        address: What a socket was asked to connect to, or a host name to look up.

    Raises:
        OSError: Naming the address, when it is a host other than this machine.
    """
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    if isinstance(address, tuple) or isinstance(address, (str, bytes)):
        if str(host) not in _LOCAL_HOST_TUPLE:
            raise OSError(errno.ENETUNREACH,
                          f"the tests reach no other machine, and this is one: {host}")


def _local_connect(self, address, *args):
    """socket.connect for the tests: a Unix socket or an address on this machine."""
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _refuse_other_machine(address)
    return _socket_connect(self, address, *args)


def _local_connect_ex(self, address, *args):
    """socket.connect_ex for the tests: a Unix socket or an address on this machine."""
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _refuse_other_machine(address)
    return _socket_connect_ex(self, address, *args)


def _local_getaddrinfo(host, *args, **kwargs):
    """socket.getaddrinfo for the tests: the names of this machine only."""
    if host is not None:
        _refuse_other_machine(host)
    return _socket_getaddrinfo(host, *args, **kwargs)


# No test reaches another machine: a connection or a name lookup that would is refused
# in this process before anything is sent
socket.socket.connect = _local_connect
socket.socket.connect_ex = _local_connect_ex
socket.getaddrinfo = _local_getaddrinfo
# litellm reads the model table and the table of Anthropic beta headers it ships with
# rather than fetching them, whatever this machine has set, and no request is handed
# to a proxy or checked against a file of certificate authorities this machine names:
# all of it is set before the package, and litellm with it, is read
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"] = "True"
_ROUTE_NAME_TUPLE = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy",
                     "https_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE",
                     "REQUESTS_CA_BUNDLE")

# The names the environment of whoever runs the tests is kept clear of: the ones of
# _ROUTE_NAME_TUPLE, and, once the package is read below, git's
_KEEP_OUT_NAME_TUPLE = _ROUTE_NAME_TUPLE
for _name in _KEEP_OUT_NAME_TUPLE:
    os.environ.pop(_name, None)

# The .env file of the directory the tests are started in sets none of
# _KEEP_OUT_NAME_TUPLE, whenever config reads it: at the import below, at a reload a
# test or a fixture makes, and at the reload after every test. A .env file elsewhere is
# read as config reads it
_START_ENV_FILE = os.path.abspath(".env")
_load_dotenv = dotenv.load_dotenv


def _load_dotenv_without_route(dotenv_path=None, *arg_tuple, **kwarg_dict):
    """dotenv.load_dotenv for the tests: the names of _KEEP_OUT_NAME_TUPLE are left as
    they were when the file is _START_ENV_FILE."""
    keep_dict = {name: os.environ.get(name) for name in _KEEP_OUT_NAME_TUPLE}
    is_start_file = (dotenv_path is not None
                     and os.path.abspath(dotenv_path) == _START_ENV_FILE)
    try:
        return _load_dotenv(dotenv_path, *arg_tuple, **kwarg_dict)
    finally:
        if is_start_file:
            for name, value in keep_dict.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


dotenv.load_dotenv = _load_dotenv_without_route

from codetwine_loom import config, guide, integrate, sandbox, tokencount, websearch  # noqa: E402
from codetwine_loom.index import store                                  # noqa: E402
from codetwine_loom.index.build import bm25_doc_dict, file_bm25_doc_dict  # noqa: E402
from codetwine_loom.index.indexer import build_bm25                     # noqa: E402
from codetwine_loom.knowledge import Knowledge, read_codetwine_setting_again  # noqa: E402
from codetwine_loom.search import query                                 # noqa: E402
from codetwine_loom.search import scope as scope_mod                    # noqa: E402
from codetwine_loom.tools import edit, explore, source, web             # noqa: E402

from knowledge_writer import write_knowledge_db                 # noqa: E402

from codetwine_loom import gitcmd                                       # noqa: E402

# The environment of whoever runs the tests names git no program and no directory of
# programs, no repository, working tree, index or object store, no file "git config"
# alone reads, and nothing that changes what git answers, so a git a test starts works
# on the repository it names
_KEEP_OUT_NAME_TUPLE = (*_ROUTE_NAME_TUPLE, *gitcmd.COMMAND_ENV_TUPLE,
                        *gitcmd.DIR_ENV_TUPLE, *config.GIT_PLACE_NAME_TUPLE,
                        *config.GIT_ANSWER_NAME_TUPLE)
for _name in _KEEP_OUT_NAME_TUPLE:
    os.environ.pop(_name, None)

# The directories programs are looked for in keep only the absolute entries of whoever
# runs the tests: an empty, "." or relative entry would name a directory of the
# repository a test makes
if "PATH" in os.environ:
    os.environ["PATH"] = os.pathsep.join(
        entry for entry in os.environ["PATH"].split(os.pathsep) if os.path.isabs(entry))

# The repositories the integration tests copy into tmp_path. Their own test files are
# not this suite's
collect_ignore = ["data"]


# The settings on by default that every test starts with off
_OFF_BY_TEST_TUPLE = ("OVERVIEW", "TEST_CREATE", "REFACTOR", "TEST_REFACTOR")


@pytest.fixture(autouse=True)
def _output_dir_of_its_own(tmp_path, monkeypatch):
    """Point OUTPUT_DIR at this test's own directory, for every test.

    OUTPUT_DIR defaults to ./out, which is where a real index and a real
    edit_history.json live. Any test that reaches history.record() - and the run_edit
    tests do, whether or not they care about the history - writes into OUTPUT_DIR, and
    pointing it here keeps those writes out of the user's ./out.

    The other settings cli.main() writes for a run, and the index it loads into the
    search module, are pinned the same way, so a test that drives main() leaves them as
    they were for the tests after it, whatever order the tests run in.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    # Each test keeps its own record of what the command line, a .env file and config
    # itself wrote over the shell's values (config.shell_env)
    for name in ("_OPTION_FILL_DICT", "_ENV_FILE_FILL_DICT", "_OWN_FILL_DICT"):
        monkeypatch.setattr(config, name, {})
    for name in ("PROJECT_DIR", "EDIT_DRY_RUN", "EDIT_GUIDE", "EDIT_HISTORY"):
        monkeypatch.setattr(config, name, getattr(config, name))
    for name in ("INDEX", "NEIGHBOR_DICT", "OUT_ADJ_DICT", "IN_ADJ_DICT",
                 "PROJECT_NAME"):
        monkeypatch.setattr(scope_mod, name, getattr(scope_mod, name))
    monkeypatch.setattr(query, "OVERVIEW_TEXT", "")
    # The token counts are this process's, so every test starts them at 0
    tokencount.reset()
    # So are what was put into a working tree and the commit a working tree made
    sandbox.reset_land_list()
    sandbox.forget_start_tree()
    # The settings of _OFF_BY_TEST_TUPLE are off, in config and in the environment
    for name in _OFF_BY_TEST_TUPLE:
        monkeypatch.setenv(name, "false")
        monkeypatch.setattr(config, name, False)


# The environment this process was started with. Every test is followed by a check
# against it (pytest_runtest_teardown below)
_ENVIRON_AT_START = dict(os.environ)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    """Put the environment back after every test, and config with it.

    A test that has config read a .env file has load_dotenv write into os.environ
    behind monkeypatch's back, so what monkeypatch puts back is not enough: the names
    the file set would stay for every test after it. This runs once every fixture,
    monkeypatch included, has been torn down and the working directory is back, so the
    reload reads what this process was started with.
    """
    result = yield
    if os.environ != _ENVIRON_AT_START:
        os.environ.clear()
        os.environ.update(_ENVIRON_AT_START)
        importlib.reload(config)
    return result


@pytest.fixture(autouse=True)
def _the_environment_is_the_test_s_own():
    """Put the environment back after every test: a test that drives cli.main() over a
    .env leaves the names of that file in os.environ, and in the settings codetwine
    holds, which are read again from the environment put back."""
    before_dict = dict(os.environ)
    yield
    if dict(os.environ) == before_dict:
        return
    for name in set(os.environ) - set(before_dict):
        del os.environ[name]
    for name, value in before_dict.items():
        if os.environ.get(name) != value:
            os.environ[name] = value
    read_codetwine_setting_again()


@pytest.fixture(autouse=True)
def _the_answers_about_the_documents_are_the_test_s_own(monkeypatch):
    """A question about the design documents answered in one test is asked again in the
    next: each test has the answers of its own."""
    from codetwine_loom import knowledge
    monkeypatch.setattr(knowledge, "_DOC_ANSWER_DICT", {})


@pytest.fixture(autouse=True)
def _a_stop_is_the_test_s_own(monkeypatch):
    """A stop made in one test stops nothing in the next: each test has a stop flag of
    its own, and the handlers cli.main() sets for SIGTERM and SIGHUP are put back."""
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    before_dict = {name: signal.getsignal(getattr(signal, name))
                   for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)}
    yield
    for name, handler in before_dict.items():
        signal.signal(getattr(signal, name), handler)


@pytest.fixture(autouse=True)
def _the_call_is_made_here(monkeypatch):
    """Leave LLM_BROKER empty for every test, whatever this machine has set.

    With an address in it every make_llm() call goes to a broker instead of to the
    litellm the tests substitute. test_broker.py and test_llm.py set it themselves for
    the tests that are about the broker.
    """
    monkeypatch.setattr(config, "LLM_BROKER", "")


@pytest.fixture(autouse=True)
def _no_other_test_dir(monkeypatch):
    """Take TEST_PATH alone for the tests' directory unless a test sets the others: what
    config computed at import depends on the directory the tests run from."""
    monkeypatch.setattr(config, "TEST_PATH_ALT_TUPLE", ())


@pytest.fixture(autouse=True)
def _no_static_check(monkeypatch):
    """Run no static check unless a test sets one up: the default command of the
    primary language would otherwise run over whatever PROJECT_DIR is."""
    monkeypatch.setattr(edit, "run_check", lambda: None)


@pytest.fixture(autouse=True)
def _the_prompt_is_one_string(monkeypatch):
    """Send every stage's prompt as one string, which is what the stand-in LLMs read;
    the tests of the block form set PROMPT_CACHE themselves."""
    monkeypatch.setattr(config, "PROMPT_CACHE", False)


@pytest.fixture(autouse=True)
def _a_key_is_set(monkeypatch):
    """Give every test a key, whatever this machine has set.

    cli.main() and sandbox.run() stop before they start when the model's provider has
    nothing to call with, and no test here calls one. The tests about that check set
    the key to "" themselves.
    """
    monkeypatch.setattr(config, "LLM_API_KEY", "a-stand-in-key")


@pytest.fixture(autouse=True)
def _forget_what_this_process_learnt(monkeypatch):
    """Start every test with an empty record of the fillings that failed and of the
    directories where a build refused the old JDK: both are kept for the whole process."""
    monkeypatch.setattr(sandbox, "_fill_fail_count_dict", collections.Counter())
    monkeypatch.setattr(edit, "_old_jdk_refuse_dir_set", set())


@pytest.fixture(autouse=True)
def _reset_patch_rounds():
    """Reset the patch rounds to 0 before every test.

    The round count in tools/edit.py is a module variable, so without the reset what
    one test spent carries into the next. This mirrors create_agent() calling
    reset_patch_round() once per request in production.
    """
    edit.reset_patch_round()


@pytest.fixture(autouse=True)
def _reset_the_instruction_file():
    """Forget the repository's instruction file before every test.

    guide.py caches what it read for one run, so without the reset a test that points
    EDIT_GUIDE at a file of its own leaves that text in the instructions every later
    test builds. This mirrors create_agent() calling guide.reset() once per request.
    """
    guide.reset()


@pytest.fixture(autouse=True)
def _reset_the_stage_record():
    """Empty what a stage recorded and read before every test.

    integrate.py keeps the decisions, the cases and the files read of one stage in a
    module variable, and every stage empties it when it starts. A test that builds a
    stage's state itself starts from what the test before it left there otherwise.
    """
    integrate.reset_stage()


@pytest.fixture(autouse=True)
def _reset_call_counters():
    """Reset the search_again, explore and web_search counters to 0 before every test,
    and the moment the next search may be made.

    They are module variables, so without the reset what one test spent carries into
    the next. This mirrors what create_agent() does in production.
    """
    source.reset_search_count()
    explore.reset_explore_count()
    web.reset_web_search_count()
    web.reset_web_fetch_count()
    websearch._not_before_dict.clear()
    websearch._hold_until_dict.clear()


@pytest.fixture(autouse=True)
def _no_web_endpoint_from_the_environment(monkeypatch):
    """Start every test with no search or page-reading endpoint and no broker.

    config reads them from the environment of whoever runs the tests, and a test that
    counts the tools or reads the instructions depends on their being unset. A test that
    wants one sets it itself.
    """
    for name in ("WEB_SEARCH_URL", "WEB_SEARCH_PROVIDER", "WEB_FETCH_URL",
                 "WEB_FETCH_PROVIDER", "LLM_BROKER"):
        monkeypatch.setattr(config, name, "")


# A self-signed certificate, valid for a hundred years
CERTIFICATE_PEM = """-----BEGIN CERTIFICATE-----
MIIBgjCCASmgAwIBAgIUXnYllWsmFFIjhtx0cP6pLVRj94wwCgYIKoZIzj0EAwIw
FjEUMBIGA1UEAwwLZ2xhLXRlc3QtY2EwIBcNMjYxMDEzMTQ0MjE3WhgPMjEyNjA5
MTkxNDQyMTdaMBYxFDASBgNVBAMMC2dsYS10ZXN0LWNhMFkwEwYHKoZIzj0CAQYI
KoZIzj0DAQcDQgAEYCSUCC9xPb4uWRUu6Msx2JN4nb9yugtQ73PCRWPsTXsbYL/S
+ruXi1e352z01hTa/pyzoeE0aL0l+Uu4OH1zw6NTMFEwHQYDVR0OBBYEFH/uutQi
gAzOuP6eg4keq97J3JThMB8GA1UdIwQYMBaAFH/uutQigAzOuP6eg4keq97J3JTh
MA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDRwAwRAIgVROQ1KNyGIOtw2dr
MXdnugTH+z41snms+Y/rUTzEJrwCIAar0Ywy5xNN8B12A+kI0uQI8YexxwP4tyl3
T96MKuf8
-----END CERTIFICATE-----
"""


def git_repo(path, commit: bool = True) -> str:
    """Make a git repository at a path and return it as a string.

    Args:
        path: Where the repository goes. It is made when it is not there.
        commit: True writes one file and commits it.

    Returns:
        The path, as a string.
    """
    path.mkdir(parents=True, exist_ok=True)
    for args in (("init", "--quiet", "-b", "main"),
                 ("config", "user.email", "t@example.invalid"),
                 ("config", "user.name", "t")):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    if commit:
        (path / "a.py").write_text("x = 1\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(path), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(path), "commit", "--quiet", "-m", "first"],
                       check=True, capture_output=True)
    return str(path)


def _definition(name: str, type_: str, start: int, end: int, context: str) -> dict:
    """Build one definitions[] entry of the analysis."""
    return {"name": name, "type": type_, "start_line": start, "end_line": end,
            "context": context}


@pytest.fixture
def knowledge_of(tmp_path):
    """Hand out the reader the index build takes, built from a synthetic analysis.

    The dict goes through the same database the tool reads in production, so a defect
    on that path shows up here too. Every reader handed out is closed at the end of the
    test.

    Yields:
        make(pk) -> Knowledge. It may be called more than once per test; each call
        writes its own database.
    """
    opened = []

    def make(pk: dict) -> Knowledge:
        path = str(tmp_path / f"knowledge_{len(opened)}.sqlite")
        write_knowledge_db(pk, path, pk.get("project_name", "p"))
        knowledge = Knowledge.open(path)
        opened.append(knowledge)
        return knowledge

    yield make
    for knowledge in opened:
        knowledge.close()


@pytest.fixture
def java_pk() -> dict:
    """Synthetic Java data: a Config class plus 2 constructors of the same name.

    class_declaration "Config" and constructor_declaration "Config" produce the same
    key, so unless step 1 splits them with #<type>#<start line> the class's node
    disappears.
    """
    return {
        "project_name": "app",
        "files": [{
            "file": "app/src/main/java/com/example/Config_java/Config.java",
            "file_dependencies": {
                "definitions": [
                    _definition("Config", "class_declaration", 3, 20,
                                "public class Config {"),
                    _definition("Config", "constructor_declaration", 5, 7,
                                "public Config() {"),
                    _definition("Config", "constructor_declaration", 9, 11,
                                "public Config(int timeout) {"),
                    _definition("getApiBase", "method_declaration", 13, 15,
                                "public static String getApiBase() {"),
                    _definition("API_BASE", "field_declaration", 4, 4,
                                'private static final String API_BASE = "";'),
                ],
                "callee_usages": [],
                "caller_usages": [],
            },
            "doc": {
                "sections": [
                    {"id": "overview", "title": "Overview",
                     "content": "`Config` holds the settings for the client.\n\n"
                                "The constant `API_BASE` is the default endpoint."},
                    {"id": "definitions", "title": "Definitions",
                     "content": "## `Config`\nThe configuration holder and its "
                                "constructors.\n\n"
                                "## `getApiBase`\nReturns the configured base URL."},
                ],
                "summary": "**Config.java** holds client settings.\nSecond line.",
            },
        }],
    }


def _def_node(name: str, file: str, lines: list[int], signature: str,
              doc: str) -> dict:
    """Build one nodes.defs entry with the same 6 keys as build_def_node_dict()."""
    return {"name": name, "type": "function_definition", "file": file,
            "lines": lines, "signature": signature, "doc": doc}


def _built(defs: dict, files: dict, index_def: dict, index_file: dict,
           calls: list, path: str, project: str = "p",
           commit: str = "") -> store.Index:
    """Write one index the way step 5 writes it and open it for reading.

    Args:
        defs: The definition nodes.
        files: The file nodes.
        index_def: The definition index, as build_bm25() returns it.
        index_file: The file index, likewise.
        calls: [(calling key, called key), ...].
        path: Where to write the index.
        project: The project name to record in the index.
        commit: The commit to record as the one the analysis describes.

    Returns:
        The Index the search side reads through.
    """
    idx = {"nodes": {"defs": defs, "files": files},
           "edges": {"calls": list(calls)},
           "index": {"def": index_def, "file": index_file}}
    store.save(path, idx, defs, "1:2026-01-01T00:00:00+00:00", project, commit)
    return store.load(path)


def scoring_index(docs: dict, path: str, types: dict | None = None,
                  names: dict | None = None) -> store.Index:
    """Build a definition index whose searchable text is given directly.

    Going through store.save() and store.load() is what production does; scoring
    against build_bm25()'s own output would be a shape production never sees.

    Args:
        docs: {definition key: the searchable text}. It is given to build_bm25() as the
            code half of that definition's text, the half the language's keywords are
            taken out of.
        path: Where to write the index.
        types: {definition key: its tree-sitter type}, for the keys that are not to be
            plain functions. None makes every definition a function.
        names: {definition key: the names that definition holds}, for the keys that
            hold any. None makes no definition hold a name.

    Returns:
        The Index bm25_rank() scores against.
    """
    defs = {}
    for key in docs:
        file, _, name = key.rpartition("::")
        defs[key] = _def_node(name, file, [1, 2], "", "")
    for key, type_ in (types or {}).items():
        defs[key]["type"] = type_
    doc_dict = {key: (text, "", (names or {}).get(key, [])) for key, text in docs.items()}
    return _built(defs, {}, build_bm25(doc_dict), build_bm25({}), [], path)


def write_index(path: str, defs: dict, files: dict,
                project: str = "p", commit: str = "") -> store.Index:
    """Write one index at a given path, for a test that needs it on disk.

    Args:
        path: Where to write the index.
        defs: The definition nodes.
        files: The file nodes.
        project: The project name to record in the index.
        commit: The commit to record as the one the analysis describes.

    Returns:
        The Index the search side reads through.
    """
    return _built(defs, files, build_bm25(bm25_doc_dict(defs, project)),
                  build_bm25(file_bm25_doc_dict(files, project)), [], path, project,
                  commit)


def fake_defs() -> dict:
    """3 definition nodes over 2 files, two of them sharing a name."""
    return {
        "p/cfg_py/cfg.py::load_config": _def_node(
            "load_config", "p/cfg_py/cfg.py", [10, 42],
            "def load_config(path):", "reads the configuration file"),
        "p/cfg_py/cfg.py::save_config": _def_node(
            "save_config", "p/cfg_py/cfg.py", [45, 60],
            "def save_config(path, data):", "writes the configuration file"),
        "p/util_py/util.py::load_config": _def_node(
            "load_config", "p/util_py/util.py", [1, 5],
            "def load_config():", "unrelated helper"),
    }


def fake_files() -> dict:
    """The 2 file nodes matching fake_defs(), one of them without a summary."""
    return {
        "p/cfg_py/cfg.py": {"summary": "This file reads and writes settings.",
                            "defs": ["p/cfg_py/cfg.py::load_config",
                                     "p/cfg_py/cfg.py::save_config"]},
        "p/util_py/util.py": {"summary": "",
                              "defs": ["p/util_py/util.py::load_config"]},
    }


@pytest.fixture
def index_of(tmp_path, monkeypatch):
    """Hand out a factory that writes one index and installs it as scope.IDX.

    Reading a real index would tie the tests to codetwine's output, so the minimal
    shape of one is assembled here and written through step 5.

    Yields:
        make(defs, files, calls=()) -> Index. Each call writes its own index.
    """
    opened = []

    def make(defs: dict, files: dict, calls=()) -> store.Index:
        path = str(tmp_path / f"index_{len(opened)}.sqlite")
        index = _built(defs, files, build_bm25(bm25_doc_dict(defs, "p")),
                       build_bm25(file_bm25_doc_dict(files, "p")), list(calls), path)
        opened.append(index)
        neighbours: dict[str, list[str]] = {}
        out_adj_dict: dict[str, list[str]] = {}
        in_adj_dict: dict[str, list[str]] = {}
        for a, b in calls:
            neighbours.setdefault(a, []).append(b)
            neighbours.setdefault(b, []).append(a)
            out_adj_dict.setdefault(a, []).append(b)
            in_adj_dict.setdefault(b, []).append(a)
        monkeypatch.setattr(scope_mod, "INDEX", index)
        monkeypatch.setattr(scope_mod, "NEIGHBOR_DICT", neighbours)
        monkeypatch.setattr(scope_mod, "OUT_ADJ_DICT", out_adj_dict)
        monkeypatch.setattr(scope_mod, "IN_ADJ_DICT", in_adj_dict)
        monkeypatch.setattr(scope_mod, "PROJECT_NAME", "p")
        monkeypatch.setattr(source, "project_name", "p")
        return index

    yield make
    for index in opened:
        index.close()


@pytest.fixture
def fake_index(index_of) -> store.Index:
    """A synthetic index of 3 definitions and 2 files, installed as scope.IDX.

    Used by the search_scope family (def_detail, file_summaries, search_file_list,
    rrf_merge, spread) and by read_definition and search_again.

    Returns:
        The Index that was put into scope.IDX.
    """
    return index_of(fake_defs(), fake_files())


@pytest.fixture
def js_defs() -> dict:
    """Synthetic JavaScript definition nodes: 2 constants and 1 function.

    A JavaScript constant is a lexical_declaration, so an implementation written as
    type != "expression_statement" excludes none of them.
    """
    return {
        "app/src/config_js/config.js::API_BASE": {
            "name": "API_BASE", "type": "lexical_declaration"},
        "app/src/config_js/config.js::TIMEOUT": {
            "name": "TIMEOUT", "type": "lexical_declaration"},
        "app/src/config_js/config.js::loadConfig": {
            "name": "loadConfig", "type": "function_declaration"},
    }


def pytest_configure(config):
    """Register the marker of the tests under tests/integration."""
    config.addinivalue_line(
        "markers", "integration: drives whole subcommands over a repository made in tmp_path")
