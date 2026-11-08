"""What the integration tests share: a repository to change, a model that answers
from a script, and the settings that point every subcommand at both.

The model is an HTTP server on this machine speaking the Messages API litellm posts
to, so every call the tool makes goes through litellm as it does in production.
Which reply comes back is decided by the prompt: the stage is read off its
instructions and the turn off the "## Turn i/n" line at the end.
"""

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from codetwine_loom import config, guide, image, websearch
from codetwine_loom.rlm import NEXT_TURN_TITLE
from codetwine_loom.tools import edit, explore, source, web

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data")

# The phrase at the top of each stage's instructions, which is what names the stage a
# prompt belongs to. A prompt holding none of them is recorded under "other"
_STAGE_PHRASE_DICT = {
    "expand": "Rewrite this software change request as English",
    "edit": "You change files in a",
    "plan": "You plan a change to a",
    "dispatch": "decide what is\nto be done with one request",
    "combine": "## Putting the agents' answers together",
    "answer": "You answer one question about a",
    "integrate": "You are combining changes several people made to",
    "test_case": "You decide what the tests of a change to a",
    "test_create": "You are writing tests for a change that was just",
    "refactor": "You are refactoring a change that was just",
    "test_refactor": "You are refactoring the tests of a change that was just",
    "report": "You are writing the last thing this tool prints",
    "explore": "You are an explorer",
}

# The line the RLM ends every prompt with, which says which turn is about to run
_TURN_LINE = re.compile(rf"^{re.escape(NEXT_TURN_TITLE)} (\d+)/(\d+)$", re.M)

# What a stage that used every turn is asked with
_LATE_MARK = "## Out of turns"

# The model name every stage is pointed at. litellm reads the provider off the prefix and
# what the model takes off the name, and posts the Messages API request the server here
# answers; no call leaves this machine
MODEL = "anthropic/claude-haiku-4-5"


def code(text: str) -> str:
    """Wrap one turn's code in the fence the RLM reads."""
    return f"```python\n{text}\n```"


def complete(text: str) -> str:
    """Return the turn that ends a stage with the text given."""
    return code(f"COMPLETE({text!r})")


def complete_json(value: object) -> str:
    """Return the turn that ends a stage with the object given, as its JSON."""
    return code(f"COMPLETE({json.dumps(value, ensure_ascii=False)!r})")


class Script:
    """The replies the model gives, per stage and turn.

    Attributes:
        reply_dict: {stage: [reply for turn 1, reply for turn 2, ...]}. A turn past the
            end of the list gets the last entry again.
        late_reply: What a stage that ran out of turns answers.
        other_reply: What a prompt of no known stage gets.
        call_list: Every call made, as {"stage", "turn", "prompt", "reply"}, in order.
    """

    def __init__(self, reply_dict: dict[str, list[str]] | None = None,
                 late_reply: str = "what was done", other_reply: str = ""):
        self.reply_dict = dict(reply_dict or {})
        self.late_reply = late_reply
        self.other_reply = other_reply or complete("nothing")
        self.call_list: list[dict] = []
        self.rule_list: list[tuple[str, str, list[str]]] = []
        self._lock = threading.Lock()

    def add(self, stage: str, when: str, reply_list: list[str]) -> None:
        """Answer a stage's prompts holding one text with a reply list of their own.

        Args:
            stage: The stage, as _STAGE_PHRASE_DICT names it.
            when: The text the prompt has to hold, e.g. the task the agent was given.
            reply_list: The replies, one per turn; the last one repeats.
        """
        self.rule_list.append((stage, when, list(reply_list)))

    def _reply_list_for(self, stage: str, prompt: str) -> list[str]:
        """Return the reply list a prompt is answered from, empty for none."""
        for rule_stage, when, reply_list in self.rule_list:
            if rule_stage == stage and when in prompt:
                return reply_list
        return self.reply_dict.get(stage) or []

    def stage_of(self, prompt: str) -> str:
        """Return the stage a prompt belongs to, or "other"."""
        for stage, phrase in _STAGE_PHRASE_DICT.items():
            if phrase in prompt:
                return stage
        return "other"

    def reply_for(self, prompt: str) -> str:
        """Pick the reply for one prompt and record the call."""
        stage = self.stage_of(prompt)
        turn_match = _TURN_LINE.search(prompt)
        turn = int(turn_match.group(1)) if turn_match else 1
        reply_list = self._reply_list_for(stage, prompt)
        if _LATE_MARK in prompt:
            reply = self.late_reply
        elif reply_list:
            reply = reply_list[min(turn, len(reply_list)) - 1]
        else:
            reply = self.other_reply
        with self._lock:
            self.call_list.append({"stage": stage, "turn": turn, "prompt": prompt,
                                   "reply": reply})
        return reply

    def prompt_list(self, stage: str) -> list[str]:
        """Return the prompts one stage was given, in order."""
        return [c["prompt"] for c in self.call_list if c["stage"] == stage]

    def stage_list(self) -> list[str]:
        """Return the stage of every call, in order."""
        return [c["stage"] for c in self.call_list]


def _prompt_of(content) -> str:
    """Return the text of one message's content, given as a string or as blocks."""
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


class _Handler(BaseHTTPRequestHandler):
    """Answer one Messages API request from the server's script."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request_dict = json.loads(self.rfile.read(length))
        prompt = _prompt_of(request_dict["messages"][-1]["content"])
        reply = self.server.script.reply_for(prompt)
        if request_dict.get("stream"):
            return self._stream(request_dict.get("model", ""), reply)
        body = json.dumps({
            "id": "scripted", "type": "message", "role": "assistant",
            "model": request_dict.get("model", ""),
            "content": [{"type": "text", "text": reply}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, model: str, reply: str) -> None:
        """Answer a request that asks for a stream with the reply as one text delta."""
        event_list = [
            ("message_start", {"message": {
                "id": "scripted", "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1}}}),
            ("content_block_start", {"index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0,
                                     "delta": {"type": "text_delta", "text": reply}}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps({'type': name, **field_dict})}\n\n"
                       for name, field_dict in event_list).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # A reply that ends early has the stream closed before it is all written
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def log_message(self, *arg_tuple):
        pass


@pytest.fixture
def script() -> Script:
    """The script the model answers from. A test fills reply_dict before the run."""
    return Script()


@pytest.fixture
def model_server(script, tmp_path, monkeypatch):
    """Serve the script over HTTP and point every setting at it, here and in children.

    The settings go into the environment, where a child process reads them, and config
    is read again so that this process reads the same.

    Yields:
        The Script the server answers from.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.script = script
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    setting_dict = {
        "LLM_API_KEY": "scripted-key",
        "LLM_API_BASE": f"http://127.0.0.1:{port}",
        "MEDIUM_MODEL": MODEL,
        "HIGH_MODEL": MODEL,
        "OUTPUT_DIR": str(tmp_path / "out"),
        "TEST_COMMAND": f"{sys.executable} -m pytest -q -p no:cacheprovider",
        "RUN_COMMAND": sys.executable,
        "RLM_MAX_TURN": "6",
        "TWO_STAGE": "false",
        "LLM_MAX_RETRY": "0",
        "TURN_LOG_KEEP": "5",
        "OVERVIEW": "false",
        "TEST_CREATE": "false",
        "REFACTOR": "false",
        "TEST_REFACTOR": "false",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    for name, value in setting_dict.items():
        monkeypatch.setenv(name, value)
    for name in ("LLM_BROKER", "PROJECT_DIR", "PRESET",
                 "WEB_SEARCH_PROVIDER", "WEB_FETCH_PROVIDER",
                 "CONTAINER_ENGINE", "EDIT_DRY_RUN", "EDIT_GUIDE",
                 "THINK_LEVEL", "TEST_CREATE_MAX_DEFINITION", "REFACTOR_GUIDE",
                 "TEST_GUIDE", "TEST_REFACTOR_GUIDE"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config)
    reset_counter()
    yield script
    server.shutdown()
    server.server_close()


@pytest.fixture
def host_engine(model_server, tmp_path, monkeypatch):
    """Point CONTAINER_ENGINE at the engine that runs the container's scripts here.

    Yields:
        The path of the log every run of the engine appends to.
    """
    bin_dir = tmp_path / "engine-bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "podman"
    wrapper.write_text(
        f"#!/bin/sh\nexec {sys.executable} "
        f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'host_engine.py')} "
        f'"$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    log_path = tmp_path / "engine.jsonl"
    monkeypatch.setenv("CONTAINER_ENGINE", str(wrapper))
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_LOG", str(log_path))
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_WORK", str(tmp_path / "engine-work"))
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_SOURCE", image.source_mark(image.source_dir()))
    monkeypatch.delenv("CTLOOM_HOST_ENGINE_IMAGE_FILE", raising=False)
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    monkeypatch.setenv("DISPATCH_MAX_RETRY", "0")
    importlib.reload(config)
    yield log_path


def engine_run_list(log_path) -> list[dict]:
    """Return every run the engine logged, oldest first."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def reset_counter() -> None:
    """Put the per-request counters back to 0, as create_agent() does before a run."""
    edit.reset_patch_round()
    guide.reset()
    source.reset_search_count()
    explore.reset_explore_count()
    web.reset_web_search_count()
    web.reset_web_fetch_count()
    websearch._not_before_dict.clear()
    websearch._hold_until_dict.clear()


def git(directory, *arg_tuple) -> str:
    """Run one git command in a repository and return its output."""
    proc = subprocess.run(["git", "-C", str(directory), *arg_tuple], capture_output=True,
                          text=True, check=False)
    assert proc.returncode == 0, f"git {' '.join(arg_tuple)}: {proc.stderr}"
    return proc.stdout


def commit_all(directory, message: str) -> str:
    """Stage everything, commit it and return the commit."""
    git(directory, "add", "-A")
    git(directory, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
        "commit", "-q", "--allow-empty", "-m", message)
    return git(directory, "rev-parse", "HEAD").strip()


def make_repo(target, fixture_name: str = "calc"):
    """Copy one fixture repository to target and commit it on main.

    Args:
        target: Where the repository goes.
        fixture_name: The directory under tests/data to copy.

    Returns:
        The repository's path.
    """
    shutil.copytree(os.path.join(_DATA_DIR, fixture_name), target)
    git(target, "init", "-q", "-b", "main")
    git(target, "config", "user.email", "t@example.invalid")
    git(target, "config", "user.name", "t")
    commit_all(target, "first")
    return target


@pytest.fixture
def calc_repo(tmp_path):
    """The calc fixture, committed on main under tmp_path."""
    return make_repo(tmp_path / "calc")


@pytest.fixture
def out_dir(tmp_path):
    """The output directory the settings point at."""
    return tmp_path / "out"


def branch_list(directory) -> list[str]:
    """Return the repository's branch names, sorted."""
    return sorted(git(directory, "for-each-ref", "--format=%(refname:short)",
                      "refs/heads").split())


# == The replies the calc fixture is changed with ============
SUB_TEXT = '''    return a + b


def sub(a: float, b: float) -> float:
    """Return a minus b."""
    return a - b
'''

# One agent's run: add sub() beside add(), run the tests, and finish
ADD_SUB_REPLY_LIST = [
    code('print(replace_in_file("calc/basic.py", "    return a + b\\n", '
         + repr(SUB_TEXT) + "))"),
    code("print(run_test())"),
    complete("added sub() to calc/basic.py; the tests passed"),
]
