# codetwine-loom

A command-line tool that changes code from a request written in plain language, or answers questions about the code. It builds a search index from the [codetwine](https://github.com/yumeiriowl/codetwine) analysis of a repository, finds the files and definitions the request is about, and has an LLM agent rewrite them and run the tests.

- **Search without an LLM in the index**: the index uses BM25 and the call relations codetwine extracts. The request is first turned into English search terms; it can be written in any language
- **Its own agent loop (RLM)**: the model writes Python, which runs in a REPL. No external agent framework is used
- **Changes in a container**: `sandbox` changes a clone in a container with no network and no API key, and applies the result to your working tree without committing
- **Request dispatch**: `run` splits a question into sub-questions or a change into tasks, runs agents in parallel and merges the results. After the merge, it writes tests and refactors the change
- **Many languages**: every text file is searchable. Definitions and call relations cover the 15 languages codetwine analyses (Python, Java, Kotlin, JavaScript, TypeScript, C, C++, Rust, C#, SQL, COBOL, BMS, JCL, CICS CSD and R)

## Table of Contents

- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [1. Installation](#1-installation)
  - [2. Configuration](#2-configuration)
  - [3. Usage](#3-usage)
- [🤖 Using it from a coding agent](#-using-it-from-a-coding-agent)
- [📖 Subcommands](#-subcommands)
- [⚙️ Settings](#️-settings)
  - [The minimal settings and `PRESET`](#the-minimal-settings-and-preset)
  - [All settings](#all-settings)
- [🔄 Processing flow](#-processing-flow)
- [📁 Output files](#-output-files)
- [🐳 Running in containers](#-running-in-containers)
- [🐍 Using it as a library](#-using-it-as-a-library)
- [⚠️ Limitations and notes](#️-limitations-and-notes)
- [🏗️ Project structure](#️-project-structure)
- [🙏 Acknowledgments](#-acknowledgments)
- [📄 License](#-license)

## 🚀 Quick Start

> **⚠️ Use `sandbox` or `run`.** The REPL is not a sandbox: the Python the model writes runs inside this tool's process, with your permissions ([details](docs/limitations.md#the-repl-is-not-a-sandbox)). Run `edit` directly on the host only on a repository you can afford to lose.

### Prerequisites

- `git`. The target repository must be a git repository
- Python 3.11 or later and [uv](https://docs.astral.sh/uv/) 0.11.8 or later
- An API key for the LLM. Calls go through litellm, and the prefix of the model name selects the provider
- [Podman](https://podman.io/) or [Docker](https://www.docker.com/) for `sandbox`, `run`, `integrate` and `warm`

### 1. Installation

```bash
git clone https://github.com/yumeiriowl/codetwine-loom.git
cd codetwine-loom
uv sync --locked
source .venv/bin/activate     # or call .venv/bin/codetwine-loom directly
```

Keep the target repository outside this repository. Installing with plain pip and following codetwine to its latest commit are described in [docs/usage.md](docs/usage.md#installation).

The commands that start containers use the image `codetwine-loom`, which holds the toolchains of every language. When it is missing or stale, the first command that needs it offers to build it (this takes a few minutes, needs a network and uses several GB of disk).

### 2. Configuration

Settings are environment variables, or a `.env` in the directory you run the command from. On first use, just start a subcommand that calls a model: when there is no `.env` and you are at a terminal, a guide asks for the settings, writes `.env` and carries on. `codetwine-loom setup` asks the same questions at any time ([details](docs/usage.md#the-first-run-guide-and-setup)). You can also copy the template or set the minimum in the shell:

```bash
cp codetwine_loom/.env.example .env   # then write LLM_API_KEY
# or
export LLM_API_KEY="..."              # the key for the models
export PRESET=medium                  # low / medium / high (medium when omitted)
```

The default models are `anthropic/claude-sonnet-5-5` (`MEDIUM_MODEL`) and `anthropic/claude-opus-5-5` (`HIGH_MODEL`). At start-up, subcommands that call a model check that each model has its key.

### 3. Usage

```bash
# Build the analysis and the search index (no LLM; add --doc-create for design documents, which calls the LLM)
codetwine-loom build --project /path/to/repo --output-dir ./out

# Show the candidate files and definitions for a request (changes nothing)
codetwine-loom search --project /path/to/repo --output-dir ./out \
  --request "Keep loading the settings from crashing"

# Make one change in a container and apply it to the working tree
codetwine-loom sandbox --project /path/to/repo --output-dir ./out \
  --request "Keep loading the settings from crashing"

# Answer a question, or split a change into tasks, run them in parallel, merge them, write tests and refactor
codetwine-loom run --project /path/to/repo --output-dir ./out \
  --request "Tidy up how the settings are loaded, and update the callers to match"

# Read the request from a file (Markdown is fine; - reads the standard input)
codetwine-loom run --project /path/to/repo --output-dir ./out --request-file request.md

# Carry on from an earlier run (its last line prints the names to pass)
codetwine-loom run --project /path/to/repo --output-dir ./out \
  --resume codetwine-loom/run/20260906-184500-1-1234/1

# List earlier runs and changes, with the command to carry on from each
codetwine-loom history --project /path/to/repo --output-dir ./out
```

A successful run applies its change to the working tree and commits nothing; `HEAD` and the branch do not move. Check the change with `git diff` and commit it yourself. The last line of the output is a command that takes the change back out ([example output](docs/usage.md#running-in-the-sandbox)).

## 🤖 Using it from a coding agent

`skills/codetwine-loom/` is an [Agent Skill](https://agentskills.io): a `SKILL.md` with `name` / `description` front matter, with the details in `references/`. A coding agent that has it can install and configure this tool, build the container image, run requests, read the results, carry on from failed runs and write the [instruction files](docs/settings.md#the-instruction-files-for-changes-codetwine_loommd).

Copy it into the personal skills directory of your agent:

```bash
git clone https://github.com/yumeiriowl/codetwine-loom.git

# Claude Code
mkdir -p ~/.claude/skills && cp -r codetwine-loom/skills/codetwine-loom ~/.claude/skills/

# GitHub Copilot (VS Code agent mode, Copilot CLI)
mkdir -p ~/.copilot/skills && cp -r codetwine-loom/skills/codetwine-loom ~/.copilot/skills/
```

| Agent | Personal skills directory | Project skills directory |
|------|------|------|
| Claude Code | `~/.claude/skills/` | `.claude/skills/` |
| GitHub Copilot (VS Code agent mode, Copilot CLI) | `~/.copilot/skills/` or `~/.agents/skills/` | `.github/skills/`, `.claude/skills/` or `.agents/skills/` |
| GitHub Copilot cloud agent | None; it uses only the skills committed to the repository | `.github/skills/`, `.claude/skills/` or `.agents/skills/` |
| Other agents that read Agent Skills | Their personal skills directory | Their project skills directory |

For codetwine-loom, a skill in a project skills directory of the target repository is part of that repository: runs copy it and can change it like any other file. Use the personal directory unless the repository is meant to hold the skill. Start a new session of the agent after copying (`/reload-skills` in Claude Code and `/skills reload` in Copilot CLI load it in the current one), then ask it to use codetwine-loom.

## 📖 Subcommands

| Subcommand | What it does |
|------|------|
| `build` | Makes the analysis and the index (`--doc-create` also has codetwine write design documents) |
| `search` | Shows the candidate files and definitions for a request. Changes nothing |
| `run` | Reads the request in a container. A question is split into sub-questions whose answers are combined. A change is split into tasks that agents run in parallel; the results are merged, given tests, refactored and applied to the working tree |
| `answer` | Answers one question on the host. Changes nothing |
| `edit` | Makes a change directly on the host |
| `sandbox` | Makes a change in a container and applies the result to the working tree |
| `integrate` | Combines several branches into one in a container and applies the result to the working tree |
| `test-create`, `refactor`, `test-refactor` | The stages after a merge: write tests for the changed definitions, refactor the changed files, refactor the changed tests. `run` and `integrate` run them inside a container |
| `overview` | Prepares the analysis and the index, then makes only the overall overview |
| `warm` | Fetches the test dependencies into the package cache, in a container with a network and no keys |
| `run-tests` | Runs the test commands once (what `warm` runs inside the container) |
| `broker` | Makes the model calls of containers on the host (`sandbox`, `run` and `integrate` start it themselves) |
| `format` | Runs `FORMAT_COMMAND` on the files changed since `HEAD` |
| `history` | Lists the runs of `run`, `sandbox` and `integrate` (with the command to carry on) and the changes made by `edit` |
| `setup` | Asks for the settings, writes `.env`, and checks the container image |

[docs/usage.md](docs/usage.md) has more [examples](docs/usage.md#examples-of-the-subcommands), the [arguments](docs/usage.md#arguments), [how `run` handles a request](docs/usage.md#how-run-handles-a-request), [carrying on with `--resume`](docs/usage.md#carrying-on-from-an-earlier-run-run---resume), [exit codes](docs/usage.md#exit-codes-and-how-a-run-ends-complete) and [what the subcommands print](docs/usage.md#output-of-history-build-and-search).

## ⚙️ Settings

Every setting is an environment variable named as in the tables. A value comes from the first of: a command-line argument, the environment, `.env`, the `PRESET` column, the default. Only the `.env` in the current directory is read. **A `.env` that git tracks is not read**, so keep this tool's `.env` outside the target repository. The template is `codetwine_loom/.env.example` ([how the values are read](docs/settings.md#how-the-values-are-read)).

### The minimal settings and `PRESET`

Normally these 3 are all you need to write.

| Variable | Description | Default |
|------|------|------------|
| `LLM_API_KEY` | The API key, sent with every call. It serves one provider; for models of 2 or more providers, leave it empty and set each provider's own variable (`ANTHROPIC_API_KEY` and so on) ([details](docs/settings.md#checking-the-keys-and-the-image-at-start-up)) | `""` |
| `PROJECT_DIR` | The target repository. `--project` gives it too (`--project .` for the current directory) | None. Every subcommand except `broker` and `setup` stops without it |
| `PRESET` | `low` / `medium` / `high`. Picks one column of the table below | `medium` |

`PRESET` sets the values that raise accuracy, cost and time together. It does not set the models or the turn limits. A variable in the table that you set yourself overrides the preset for that variable.

| Variable | `low` | `medium` | `high` | What changes |
|------|-------|----------|--------|--------------|
| `THINK_LEVEL` | `off` | `low` | `medium` | How hard one call thinks |
| `AGENT_MAX_TOKEN` | `16384` | `32768` | `32768` | The output limit of one call (thinking and reply together) |
| `EXPLORE_MAX_CALL` | `3` | `4` | `5` | `explore` calls per change run |
| `SEARCH_AGAIN_MAX_CALL` | `3` | `5` | `10` | `search_again` calls |
| `WEB_SEARCH_MAX_CALL` | `10` | `15` | `20` | `web_search` calls |
| `WEB_FETCH_MAX_CALL` | `5` | `8` | `10` | `web_fetch` calls |
| `RUN_PROJECT_MAX_CALL` | `10` | `15` | `20` | `run_project` calls per stage |
| `MAX_PATCH_ROUND` | `3` | `5` | `8` | Round trips of "rewrite and test" |
| `TWO_STAGE` | `false` | `true` | `true` | The planning stage before the change |
| `DISPATCH_MAX_TASK` | `5` | `8` | `10` | Pieces `run` splits one request into |
| `DISPATCH_MAX_RETRY` | `2` | `3` | `5` | Hand-outs again of a task that did not finish |
| `DISPATCH_MAX_RERUN` | `1` | `2` | `3` | Reruns of a request that did not succeed |
| `INTEGRATE_MAX_REFUSAL` | `1` | `2` | `3` | Send-backs of the merge's `COMPLETE()` while checks remain |
| `INTEGRATE_MAX_RERUN` | `0` | `1` | `2` | Redos of an unsettled merge in `run` |
| `TEST_CREATE_MAX_RERUN` | `1` | `2` | `3` | Redos of an unsettled test-writing stage |
| `REFACTOR_MAX_RERUN` | `1` | `2` | `3` | Redos of an unsettled refactoring stage |

### All settings

Every setting has a row in [docs/settings.md](docs/settings.md), grouped as follows:

- [Start-up checks of the keys and the image](docs/settings.md#checking-the-keys-and-the-image-at-start-up), [proxies and certificates](docs/settings.md#proxies-and-certificates), [paths and analysis](docs/settings.md#paths-and-analysis)
- [Index building and search](docs/settings.md#index-building-and-search), [the overall overview](docs/settings.md#the-overall-overview)
- [Models and model calls](docs/settings.md#models-and-model-calls), [agents](docs/settings.md#agents), [dispatching](docs/settings.md#dispatching-and-the-report-run), [merging](docs/settings.md#merging-integrate), [edit history](docs/settings.md#edit-history)
- [Writing tests after the merge](docs/settings.md#writing-tests-after-the-merge-test_create), [refactoring after the merge](docs/settings.md#refactoring-after-the-merge-refactor--test_refactor), [instruction files (`CODETWINE_LOOM.md`)](docs/settings.md#the-instruction-files-for-changes-codetwine_loommd)
- [The planning stage](docs/settings.md#the-planning-stage-two-stages), [the explorer](docs/settings.md#the-explorer), [reading tools](docs/settings.md#reading-tools-read_source_file-grep_source), [`web_search`](docs/settings.md#asking-about-facts-outside-the-repository-web_search), [`web_fetch`](docs/settings.md#reading-a-page-of-the-search-results-web_fetch)
- [Change tools](docs/settings.md#change-tools), [tests and the static check](docs/settings.md#tests-and-the-static-check), [`run_project`](docs/settings.md#running-the-program-run_project), [automatic check after each edit](docs/settings.md#automatic-check-after-each-edit-smoke_arg)
- [Sandbox](docs/settings.md#sandbox), [the broker and the logs](docs/settings.md#the-broker-and-the-logs), [defaults by language](docs/settings.md#defaults-by-language)

## 🔄 Processing flow

1. **[Preparing the analysis](docs/processing-flow.md#1-preparing-the-analysis)**: when the output directory has no codetwine analysis of the current working tree, codetwine makes one. The LLM is called only for design documents (`--doc-create`)
2. **[Building the index](docs/processing-flow.md#2-building-the-index)**: `code_index.sqlite` is made from the definitions, call relations and design documents, without an LLM
3. **[Search](docs/processing-flow.md#3-search)**: the request is turned into English search terms (one LLM call), definitions are ranked with BM25, call relations are followed one hop, and the candidates are grouped by file
4. **[Changing the code](docs/processing-flow.md#4-changing-the-code)**: each turn, the model writes one Python block that runs in the REPL, for up to `RLM_MAX_TURN` turns. With `TWO_STAGE=true`, a read-only planning stage runs first
5. **[Dispatching](docs/processing-flow.md#5-dispatching-a-request-run)** (`run`): a read-only stage turns the request into sub-questions, tasks, or things to confirm with you. Tasks go to `sandbox` runs, and sub-questions to [workers whose answers are combined](docs/processing-flow.md#6-answering-questions-answer)
6. **[Merging](docs/processing-flow.md#7-merging-branches-integrate)** (`integrate`): the branches are merged one at a time. The machine checks conflicts, broken references, the tests and more, and only what is left over goes to the model
7. **After the merge**: [a stage writes tests](docs/processing-flow.md#8-writing-tests-after-the-merge-test-create) for the changed definitions, then [two stages refactor](docs/processing-flow.md#9-refactoring-after-the-merge-refactor-test-refactor) the changed code and the changed tests

## 📁 Output files

`build` writes to `<OUTPUT_DIR>/<repo name>/`, where `<repo name>` is the directory name of the target repository. `sandbox`, `run` and `integrate` give each agent its own copy under `<OUTPUT_DIR>/agents/`. The full list and the formats are in [docs/output.md](docs/output.md).

| File | Description |
|------|------|
| `<repo name>/project_knowledge.sqlite` | codetwine's analysis (definitions, call relations, design documents) |
| `<repo name>/code_index.sqlite` | The search index ([tables](docs/output.md#code_indexsqlite)) |
| `<repo name>/overview.json` | The overall overview ([format](docs/output.md#overviewjson)) |
| `<repo name>/edit_history.json` | The record of each `edit` on the host ([format](docs/output.md#edit_historyjson)) |
| `agents/.repo/<repo name>/record/<run-id>.json` | The record of one run of `run`, `sandbox` or `integrate`, which `history` lists ([format](docs/output.md#run-record-recordrun-idjson)) |
| `agents/.repo/<repo name>/land-last.diff` | The diff the last command applied to the working tree |
| `agents/<agent>/<repo name>/turns/` | That agent's log (`.log`) and the contents of each turn (`.jsonl`) |

## 🐳 Running in containers

`sandbox`, `run` and `integrate` make changes and merges on a clone inside a container ([details](docs/container.md)).

- **What goes in**: a read-only copy of the repository's git contents and a commit that copies the working tree, including uncommitted changes and untracked files that are not ignored. Files `.gitignore` ignores and the contents of submodules are not included ([the boundary](docs/container.md#the-boundary-sandbox-sets-up))
- **What stays out**: the container has no network, and the API keys are not passed to it. A broker on the host makes the model calls, `web_search` and `web_fetch`, and enforces the call limits ([the broker](docs/container.md#the-broker-model-calls))
- **Dependencies**: `warm` fetches the test dependencies into a cache, in a container with a network and no keys. When a change edits a dependency file, the cache is filled again (`WARM_MAX_REFILL`). Building dependencies runs the repository's code, so set `WARM_MAX_REFILL=0` for untrusted requests ([details](docs/container.md#fetching-the-dependencies-ahead-warm))
- **How the result comes back**: the container returns one patch. The host commits it to a branch and, when the run succeeded, applies it to the working tree. A change that touches an untracked `.env`, ignored paths, nested repositories or git's settings is not applied ([details](docs/container.md#before-a-change-goes-into-the-working-tree)). A failed run's work stays on the branch `codetwine-loom/sandbox/<run-id>` ([details](docs/container.md#what-a-run-that-did-not-pass-leaves))
- **The image**: every container runs `codetwine-loom`. To build it by hand, or to build a smaller image with some languages only, see [Building the image](docs/container.md#building-the-image)

## 🐍 Using it as a library

```python
from codetwine_loom import search_scope, create_agent, run_edit, config
from codetwine_loom.knowledge import Knowledge
from codetwine_loom.search.scope import load_index
from codetwine_loom.tools import source

# Load the index and the analysis first, as cli.py does at start-up
config.PROJECT_DIR = "../codetwine"
config.OUTPUT_DIR = "./out"
load_index("./out/codetwine/code_index.sqlite", "codetwine")
knowledge = Knowledge.open("./out/codetwine/project_knowledge.sqlite")
source.set_project_data(knowledge)
knowledge.close()
source.project_name = "codetwine"

scope = search_scope("Keep it from crashing when an error occurs while loading the settings file")
scope["files"]             # paths of the files to read (highest rank first)
scope["defs"][0]["lines"]  # [start line, end line] of that definition
scope["file_summaries"]    # {file path: summary of that file}
scope["confidence"]        # "high" or "low"
scope["spread"]            # number of files the directly hit definitions are spread over
scope["note"]              # one line when nothing was hit or step 1 failed (otherwise empty)

result = run_edit(create_agent(), "Keep loading the settings file from crashing")
result["answer"]           # the argument of COMPLETE(). None when the limit was reached
```

Set all 5 values (`config.PROJECT_DIR`, `config.OUTPUT_DIR`, `load_index()`, `source.set_project_data()`, `source.project_name`) before `search_scope()`; without them the search fails. Call `create_agent()` after `load_index()`. The return value is described in [docs/output.md](docs/output.md#return-value-of-search_scope).

## ⚠️ Limitations and notes

The limits ordinary use runs into. The full list, including the limits per language, is in [docs/limitations.md](docs/limitations.md).

- **The REPL is not a sandbox**: the code the model writes runs with this tool's permissions. The only boundary is the container of `sandbox`, `run` and `integrate` ([details](docs/limitations.md#the-repl-is-not-a-sandbox))
- **Start it with the `codetwine-loom` command**, not with `python -m codetwine_loom` from inside the target repository ([details](docs/limitations.md#do-not-start-it-with-python--m-codetwine_loom-from-inside-the-target-repository))
- **A `.env` that git tracks is not read**; when it holds this tool's settings, every subcommand stops ([details](docs/limitations.md#a-env-the-target-repository-tracks-is-not-read))
- **A repository whose directory name is `agents` cannot be handled**: clone it under another name ([details](docs/limitations.md#a-repository-whose-directory-name-is-agents-cannot-be-handled))
- **A run that ends with exit status 0 writes to the working tree**, even when the tests are red, and nothing checks that the change meets the request. The recorded verdict is never `passed`. Read the diff before you commit ([details](docs/limitations.md#a-run-that-ends-with-exit-status-0-writes-to-the-working-tree), [verdicts](docs/limitations.md#whether-a-change-is-finished-cannot-be-decided-by-the-machine), [`run_test()`](docs/limitations.md#run_test-is-not-a-passfail-gate))
- **Do not run several commands on the same working tree at once**; leave parallel work to `run` ([details](docs/limitations.md#do-not-put-runs-side-by-side-on-the-same-working-tree))
- **Ignored files are not in the container**: files `.gitignore` ignores (tests included) and the contents of submodules ([details](docs/limitations.md#the-base-is-the-working-tree-itself))
- **Design documents cost LLM calls per file**. Only the first `--doc-create` writes them for every file, and automatic rebuilds write none; run `build --doc-create` again to update them. When there is no analysis, only `run` asks whether to make them ([cost](docs/limitations.md#the-cost-of-design-documents-grows-with-the-repository), [rebuilds](docs/limitations.md#automatic-rebuilds-make-no-design-documents), [asking](docs/limitations.md#only-run-asks-when-there-is-no-analysis))
- **Cost limits**: set `LLM_MAX_CALL` or `MAX_TOKEN_PER_RUN` (both unlimited by default), or a limit at the provider. Limits kept inside the container, such as `RLM_MAX_TURN`, can be changed by the model's code ([details](docs/limitations.md#cost-limits))
- **`warm` and refills use the network** and run the repository's build settings, which a change may have edited ([details](docs/limitations.md#warm-and-refills-use-the-network))
- **The broker and the endpoint**: a `tcp://` broker has no authentication, so use `unix://` on one machine. `LLM_API_BASE` must not return redirects ([authentication](docs/limitations.md#a-tcp-broker-has-no-authentication), [redirects](docs/limitations.md#write-an-endpoint-that-does-not-return-redirects-in-llm_api_base))
- **The default tests depend on the language**: languages other than the primary one run their tests only where their dependency file is, C / C++ needs CMake or a makefile `test` target, R needs testthat's `tests/testthat`, and COBOL has no default. Otherwise set `TEST_COMMAND` ([by language](docs/limitations.md#by-language))

## 🏗️ Project structure

```text
codetwine-loom/
├── pyproject.toml
├── uv.lock                  # locked dependency versions (for .venv and the image)
├── uv.toml                  # uv settings (a 1-day waiting period for new versions)
├── Dockerfile               # the container image; TOOLCHAIN picks the languages (all by default)
├── doc_template.json        # the design-document template used when DOC_TEMPLATE_PATH is not set
├── docs/                    # the reference: usage, settings, processing flow, output, containers, limitations
├── sandbox/                 # scripts run inside the container (entry point, cache filling, default checks and tests per language)
├── skills/codetwine-loom/   # a skill that lets a coding agent set up and use this tool
├── tests/                   # pytest tests; integration/ runs the subcommands against a scripted model (uv sync --locked --extra dev)
└── codetwine_loom/
    ├── .env.example         # the settings template (the first-run guide asks for its items)
    ├── cli.py               # the subcommands
    ├── config.py            # reads the settings
    ├── knowledge.py         # prepares and reads the codetwine analysis
    ├── index/               # builds code_index.sqlite
    ├── search/              # search (request → search terms → candidates)
    ├── rlm.py               # the RLM (REPL, loop, prompt)
    ├── agent.py             # puts the instructions and tools together for a change
    ├── tools/               # the tools given to the model (reading, changing, run_project, web)
    ├── plan.py              # the planning stage
    ├── dispatch.py          # run's dispatching stage
    ├── answer.py            # answering questions and combining the answers
    ├── overview.py          # the overall overview
    ├── integrate.py         # merging branches
    ├── testcreate.py        # writing tests after the merge
    ├── refactor.py          # the refactoring stages
    ├── sandbox.py           # the host side of container runs
    ├── broker.py            # model calls, web search and cache refills for containers
    ├── image.py             # the container image's mark and build command
    └── history.py           # the record of each change
```

## 🙏 Acknowledgments

- [codetwine](https://github.com/yumeiriowl/codetwine) — analyses source code with tree-sitter and puts the definitions, call relations and design documents into `project_knowledge.sqlite`
- [litellm](https://github.com/BerriAI/litellm) — a provider-independent entry point for LLM calls
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab), [alexzhang13/rlm](https://github.com/alexzhang13/rlm), [DSPy's `dspy.RLM`](https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/rlm.py) — the idea of the RLM was drawn from these

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
