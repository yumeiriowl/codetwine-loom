# Usage

This page covers installation, the first-run guide, the subcommands and their arguments, and what the subcommands print. Every setting is listed in [settings.md](settings.md).

## Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing with uv](#installing-with-uv)
  - [Installing with plain pip](#installing-with-plain-pip)
  - [Following codetwine](#following-codetwine)
  - [The waiting period for new versions](#the-waiting-period-for-new-versions)
- [The first-run guide and `setup`](#the-first-run-guide-and-setup)
  - [`DOC_TEMPLATE_PATH`](#doc_template_path)
- [Examples of the subcommands](#examples-of-the-subcommands)
- [Arguments](#arguments)
  - [Arguments the tool passes itself](#arguments-the-tool-passes-itself)
- [Running in the sandbox](#running-in-the-sandbox)
- [How `run` handles a request](#how-run-handles-a-request)
  - [Progress lines](#progress-lines)
  - [The report at the end](#the-report-at-the-end)
- [Carrying on from an earlier run (`run --resume`)](#carrying-on-from-an-earlier-run-run---resume)
- [Exit codes and how a run ends (`COMPLETE()`)](#exit-codes-and-how-a-run-ends-complete)
- [Output of `history`, `build` and `search`](#output-of-history-build-and-search)

## Installation

### Prerequisites

You need:

- `git`. The target repository must be a git repository.
- Python 3.11 or later and [uv](https://docs.astral.sh/uv/) 0.11.8 or later. An older uv stops with `Failed to parse` or `Required uv version` when it reads this repository's `uv.toml`; upgrade it with `uv self update`.
- An API key for the LLM. Calls go through litellm, and the prefix of the model name selects the provider.
- [Podman](https://podman.io/) or [Docker](https://www.docker.com/) for `sandbox`, `run`, `integrate` and `warm`.

The `codetwine-loom` command runs on the host. The index is built on the host even when the changes are made in containers.

### Installing with uv

```bash
uv sync --locked
```

This installs the dependencies into `.venv` at the versions in `uv.lock`, and puts the `codetwine-loom` command in `.venv/bin/`. When `pyproject.toml` or `uv.toml` disagrees with `uv.lock`, nothing is installed. Packages not in `uv.lock` are removed from `.venv` (add `--inexact` to keep them).

### Installing with plain pip

`uv` is still needed to export the locked versions. Run these steps again whenever `uv.lock` changes.

```bash
req=$(mktemp)
uv export --locked --no-emit-project --no-hashes -o "$req" \
  && python3 -c 'import sys; sys.exit(sys.version_info < (3, 11) and "python3 is older than 3.11")' \
  && python3 -m venv --clear .venv && . .venv/bin/activate \
  && pip install -r "$req" \
  && pip install --no-deps -e .
rm -f "$req"
```

When `python3` is older than 3.11, the steps stop without touching `.venv`; replace both `python3` with `python3.11` or similar. pip does not check the hashes of the files it installs on this route.

### Following codetwine

To move codetwine to the tip of its `main` branch (this also updates `uv.lock`):

```bash
uv sync --upgrade-package codetwine
```

With a plain pip install, run `uv lock --upgrade-package codetwine` instead, then rerun the pip steps. Either way, the next command that starts containers asks whether to rebuild the image.

### The waiting period for new versions

When uv resolves or upgrades `uv.lock` in this repository, it skips any PyPI version published less than 1 day ago (`exclude-newer` in `uv.toml`). Installing the versions already in `uv.lock` is not affected. The setting also applies to any project under this repository without uv settings of its own, so keep the target repository outside this repository. When a new codetwine commit needs a version younger than 1 day, uv stops with `No solution found`.

To install a version that was just published, add an upper time limit for that package to `uv.toml` (UTC, just after its upload time), then upgrade it:

```toml
exclude-newer-package = { certifi = "2026-10-23T12:00:00Z" }
```

```bash
uv sync --upgrade-package certifi
uv tree --package certifi --depth 0        # check the version that went into uv.lock
```

The limit applies only to the packages listed. When a dependency of the package is also younger than 1 day, add it inside the `{ }` too. After 1 day, remove the entries and run `uv sync`; until then, the package is never upgraded past the time written. With a plain pip install, use `uv lock` instead of `uv sync` and rerun the pip steps.

## The first-run guide and `setup`

For the first use, just start a subcommand that calls a model. When the current directory has no `.env`, and the input and output are a terminal, a guide asks for the items of the template `codetwine_loom/.env.example`: the two models (`MEDIUM_MODEL`, `HIGH_MODEL`), `LLM_API_BASE`, whether to use a proxy ([Proxies and certificates](settings.md#proxies-and-certificates)), `PRESET`, whether container runs may fetch packages a change adds (`WARM_MAX_REFILL`), the web search and fetch providers, `PROJECT_DIR`, `TEST_PATH`, `DOC_TEMPLATE_PATH`, and the keys the models need. Enter keeps the template's value, and settings the shell already sets are skipped. The guide then writes `.env` (readable only by you) and carries on. Without the guide, a missing key stops the command ([Checking the keys](settings.md#checking-the-keys-and-the-image-at-start-up)).

`codetwine-loom setup` asks the same questions at any time; run it in the directory where `.env` belongs. When `.env` exists, each question shows the current value in `[ ]`. Enter keeps it and `-` removes the setting. Only the changed lines are rewritten; your own lines and comments stay. `setup` then checks the container image and offers to build it when it is missing or stale ([Building the image](container.md#building-the-image)). It exits with code 1 when a key or a proxy setting is still missing or unreadable. It refuses to run when the input or output is not a terminal, `PYTHON_DOTENV_DISABLED` is set, the `.env` is tracked by git, or the directory is not writable.

```bash
codetwine-loom setup                  # in the directory where .env is to be placed
cp codetwine_loom/.env.example .env   # or copy the template and write LLM_API_KEY by hand
```

### `DOC_TEMPLATE_PATH`

`DOC_TEMPLATE_PATH` is the template codetwine follows when it writes design documents (with `--doc-create`, or after you answer `y` to `run`'s question). When it is not set, `doc_template.json` at the top of this tool's repository is used. When design documents are to be made and the template does not exist, `run`, `sandbox` and `integrate` stop at start-up. The bundled template writes one section of prose with one heading per definition (``## `<name>` ``, or ``## `<type-name>.<name>` `` inside a class). Index building uses these headings to attach each description to its definition as search text.

## Examples of the subcommands

The subcommands and arguments are the same on the host and inside a container.

```bash
# Make the analysis (without design documents) and the index
codetwine-loom build --project ../codetwine --output-dir ./out

# Make design documents too (calls the LLM; only for files without one, changed files and files depending on them)
codetwine-loom build --project ../codetwine --output-dir ./out --doc-create

# Keep names out of the analysis. A name with / is a path from the repository root (./vendor is only the top-level vendor)
codetwine-loom build --project ../big-repo --output-dir ./out \
  --exclude vendor migrations apps/old "apps/*/gen"

# Show the candidates (changes nothing)
codetwine-loom search --project ../codetwine --output-dir ./out \
  --request "Keep loading the settings from crashing"

# A question: split into sub-questions, answered by workers, combined into one answer
codetwine-loom run --project ../codetwine --output-dir ./out \
  --request "Where are the settings loaded? What happens when the environment variable is missing?"

# A change: split into tasks, run in parallel, merged, given tests and refactored
codetwine-loom run --project ../codetwine --output-dir ./out \
  --request "Tidy up how the settings are loaded, and update the callers to match"

# The same without the overall overview and without the stages after the merge
codetwine-loom run --project ../codetwine --output-dir ./out \
  --request "Tidy up how the settings are loaded, and update the callers to match" \
  --no-overview --no-test-create --no-refactor --no-test-refactor

# Pass the request from a file (- is the standard input)
codetwine-loom run --project ../codetwine --output-dir ./out --request-file request.md
cat request.md | codetwine-loom run --project ../codetwine --output-dir ./out --request-file -

# Answer one question directly (changes nothing)
codetwine-loom answer --project ../codetwine --output-dir ./out \
  --request "Where are the settings loaded?"

# Make a change directly on the host
codetwine-loom edit --project ../codetwine --output-dir ./out \
  --request "Keep loading the settings from crashing"

# Only show the diff it would write
codetwine-loom edit --project ../codetwine --output-dir ./out --request "..." --dry-run

# Read the instruction file from another place in the repository
codetwine-loom edit --project ../codetwine --output-dir ./out --request "..." --guide docs/agent.md

# Carry on from a run that ended part-way, without asking
codetwine-loom edit --project ../codetwine --output-dir ./out --request "..." --resume

# Make a change in a container and apply it to the working tree
codetwine-loom sandbox --project ../codetwine --output-dir ./out \
  --request "Keep loading the settings from crashing"

# Combine branches into one (merged in the order given)
codetwine-loom integrate --project ../codetwine --output-dir ./out \
  --branch codetwine-loom/agent1 --branch codetwine-loom/agent2

# Fetch the dependencies into the package cache (in a container with a network)
codetwine-loom warm --project ../codetwine --output-dir ./out

# List the runs and the changes so far
codetwine-loom history --project ../codetwine --output-dir ./out

# Run the broker on its own (sandbox, run and integrate start it themselves)
codetwine-loom broker --endpoint unix://./out/llm.sock

# Run the test commands once (what warm calls inside the container)
codetwine-loom run-tests --project ../codetwine --output-dir ./out

# Prepare the analysis and the index, then make only the overall overview
codetwine-loom overview --project ../codetwine --output-dir ./out
```

`build` without `--doc-create` rebuilds only the index when the analysis matches the current working tree, and otherwise makes the analysis again without design documents. With `--doc-create`, codetwine always runs. A command that would rewrite the analysis stops while a `sandbox`, `run` or `integrate` run is using the same output directory. `search`, `answer`, `edit` and `run-tests` make a missing analysis without design documents and carry on; only `run` asks whether to make design documents ([Preparing the analysis](processing-flow.md#1-preparing-the-analysis)).

`search`, `run`, `answer`, `edit` and `sandbox` first make the overall overview (`overview.json`) when it is missing or stale, which calls the LLM ([The overall overview](settings.md#the-overall-overview)).

## Arguments

| Argument | Subcommands | Description | Default |
|------|------|------|------------|
| `--project` | all except `setup` | The target repository. Every subcommand except `broker` and `setup` stops without it. Use `--project .` for the current directory. `broker` refills the package cache only when it is given | `PROJECT_DIR` (`broker`: empty) |
| `--output-dir` | all except `setup` | Where the analysis, the index and the run records go | `OUTPUT_DIR` (`./out`) |
| `--request` | `search`, `run`, `answer`, `edit`, `sandbox`, `integrate`, `test-create`, `refactor`, `test-refactor` | The request. For `run`, a question or a change; `run` decides which. For `answer`, one question. Required for `search`, `answer`, `edit` and `sandbox`, and for `run` without `--resume`. Text that is itself a flag of the tool (such as `--dry-run`) must be written as `--request=<text>`. Text with NUL or not in UTF-8 stops the command. `sandbox`, `run` and `integrate` stop when it exceeds 131,008 bytes | — |
| `--request-file PATH` | `search`, `run`, `answer`, `edit`, `sandbox`, `integrate` | Reads the request from a UTF-8 file (`-` reads the standard input). It stops when combined with `--request`, or when the file is unreadable or empty | — |
| `--doc-create` | `build` | Always runs codetwine and has it write design documents for files without one, changed files, and files that depend on them. Calls the LLM several times per file | Not made |
| `--doc-create` | `sandbox`, `run` | Also makes design documents on the host, once at the start, as `build --doc-create` does. Without it, only `run` asks `[y/N]`, and only at a terminal when the output directory has no analysis | Not made (`run` asks) |
| `--doc-create` | `integrate` | Makes design documents when the output directory has no analysis yet | Not made |
| `--exclude NAME ...` | `build`, `sandbox`, `run`, `integrate` | Directory and file names to keep out of the analysis, added to codetwine's defaults and `BUILD_EXCLUDE`. A name without `/` matches at any depth; a name with `/` is a path from the repository root and may use `fnmatch` patterns (`./vendor`, `apps/*/gen`). The names are recorded with the analysis. `sandbox`, `run` and `integrate` stop when given names that differ from the recorded ones and print the `build --exclude ...` to run. `search`, `answer`, `edit` and `run-tests` use the recorded names | None |
| `--no-overview` | `search`, `run`, `answer`, `edit`, `sandbox` | Does not make the overall overview or pass it to search (sets `OVERVIEW=false`) | Made and passed |
| `--preset` | `run`, `answer`, `edit`, `sandbox`, `integrate`, `test-create`, `refactor`, `test-refactor` | `low` / `medium` / `high` ([`PRESET`](settings.md#the-minimal-settings-and-preset)) | `PRESET` (`medium`) |
| `--dry-run` / `--no-dry-run` | `edit`, `sandbox` | The change tools run their checks but write nothing. `edit --dry-run` prints the diff it would have written and records nothing. `sandbox --dry-run` puts nothing on a branch or in the working tree | Writes |
| `--check` / `--no-check` | `edit`, `sandbox` | Holds the change to the static check (`CHECK_COMMAND`) and to the references the analysis resolves. While new findings appear, `COMPLETE()` is sent back (up to `INTEGRATE_MAX_REFUSAL` times); findings left at the end make the verdict `failed`. Not applied with `--dry-run`, or to `run`'s agents (the merge checks the same) | Checked |
| `--guide PATH` | `edit`, `sandbox` | The instruction file for changes, relative to the repository. An empty string reads none. Inside the container, an absolute path cannot be read. `run` has no such argument; its workers read `EDIT_GUIDE` | `EDIT_GUIDE` (`CODETWINE_LOOM.md`) |
| `--resume` / `--no-resume` | `edit` | Carry on from the previous run that ended part-way, or start fresh. Without either, `edit` asks at a terminal and otherwise starts fresh. `--resume` needs `EDIT_HISTORY=true` | Asks |
| `--resume NAME` | `run`, `integrate` | Carry on from a branch an earlier run left ([Carrying on](#carrying-on-from-an-earlier-run-run---resume)). `run` accepts it several times. `integrate` takes one union branch an earlier merge left, and carries on the merge from it. When the merge on the branch settled but a later stage did not, the run carries on from that stage | — |
| `--branch NAME` | `integrate` | A branch to combine. Pass it 2 or more times; the branches are merged in that order. Not needed with `--resume` ([Combining several branches](container.md#combining-several-branches-into-one-integrate)) | None |
| `--no-test-create`, `--no-refactor`, `--no-test-refactor` | `run`, `integrate` | Skip the test-writing, refactoring or test refactoring stage after the merge (sets `TEST_CREATE` / `REFACTOR` / `TEST_REFACTOR` to `false`) | All 3 run |
| `--engine` | `sandbox`, `run`, `integrate` | The container engine | `CONTAINER_ENGINE` (empty: `podman`, then `docker`) |
| `--image` | `sandbox`, `run`, `integrate` | The image. When it is missing or stale, it is built after a question at a terminal | `CONTAINER_IMAGE`, otherwise `codetwine-loom` |
| `--endpoint` | `broker` | Where the broker listens: `unix://<path>` or `tcp://<host>:<port>` | `LLM_BROKER` |

`overview`, `warm`, `run-tests`, `format` and `history` take only `--project` and `--output-dir`. `setup` takes no arguments.

### Arguments the tool passes itself

`sandbox`, `run` and `integrate` pass these to the broker and to the runs inside the container. They are not meant for manual use.

| Argument | Subcommands | Description |
|------|------|------|
| `--here` | `run`, `integrate` | Run in this repository instead of starting a container (what the container runs inside itself). `integrate --here` runs no stage after the merge. Not allowed with `--engine`, `--image`, `--doc-create`, or the `--no-*` stage flags of `integrate` |
| `--tasks-out PATH` | `run --here` | Where to write the split |
| `--answered PATH` | `answer` | The workers' answers; the run combines them instead of answering one question |
| `--answer-out PATH` | `answer` | Where to write the answer |
| `--onto NAME`, `--onto-commit COMMIT` | `integrate --here` | The branch and commit the union is written onto |
| `--outcome-out PATH` | `integrate --here`, `test-create`, `refactor`, `test-refactor` | Where to write the stage's result |
| `--gap-in PATH` | `integrate --here` | The cases the test-writing stage recorded as not done; after the merge, cases whose tests now pass are dropped. Needs `--outcome-out` |
| `--base COMMIT` | `test-create`, `refactor`, `test-refactor` (required), `broker` | The commit the change in `HEAD` was written against. For `broker`, the commit a refill checks out |
| `--target-in PATH` | `test-create` | The files to write tests for and the tests already failing |
| `--target-only` | `test-create` | Lists the changed definitions, the failing tests and the primary language, then exits without calling a model. Not allowed with `--target-in` or `--request` |
| `--agent NAME`, `--run ID`, `--fill-ext LANGS` | `broker` | The agent and run a cache refill serves, and the languages whose default test commands it also runs. Need `--project` |
| `--token-out PATH`, `--token-left N` | `broker` | Where the broker writes its token total, and the tokens left for this run (`MAX_TOKEN_PER_RUN`) |
| `--label TEXT` | `broker` | A prefix for every line the broker logs (such as `[task 2/5] `) |

## Running in the sandbox

`sandbox` gives the container a read-only copy of the repository's git contents and a commit that copies the working tree as it is, so uncommitted changes and untracked files are visible inside. The container has no network and no API key; the broker on the host makes the model calls. The result comes back as one patch. A successful run prints:

```text
base: 3f9c1a2b4d5e6f708192a3b4c5d6e7f809a1b2c3
patch: /home/me/out/agents/20260827-213000-2-4821/target-repo/sandbox/patch-20260827-213000-2-4821.diff (2.1 KB, 3 file(s))
applied to the working tree: 3 file(s), over 3f9c1a2b4d5e. Commit them yourself
tokens: 96,412 in 14 call(s): 93,870 input (71,204 read from the cache, 9,318 written to it), 2,542 output
record: /home/me/out/agents/.repo/target-repo/record/20260827-213000-1-4821.json
to take it back out of the working tree: git -C /home/me/target-repo apply --binary --whitespace=nowarn --no-ignore-whitespace -R /home/me/out/agents/.repo/target-repo/land-last.diff
```

`base` is the commit that copies the working tree at the start (`HEAD` itself when the working tree is clean). The change is applied to the working tree; neither `HEAD` nor the branch moves, so check it with `git diff` and commit it yourself. `tokens:` is the total of the run's model calls, and `record:` is the run's record ([format](output.md#run-record-recordrun-idjson)). The last line removes the applied change again. When a run does not succeed, nothing is applied, and what it wrote stays on the branch `codetwine-loom/sandbox/<run-id>`. The container boundary, the broker and the image are described in [container.md](container.md).

## How `run` handles a request

`run` first reads the request in a read-only stage in a container. That stage only splits the request; it neither answers nor changes anything. It returns one of three results:

- **A question** (`{"questions": [...]}`): one worker per sub-question writes an answer (up to `DISPATCH_PARALLEL` at once, with reading tools only). A merge run then combines the answers, reading the code only where they disagree or leave gaps. A sub-question without an answer is handed out again, up to `DISPATCH_MAX_RETRY` times. If the merge still returns nothing after `INTEGRATE_MAX_RERUN` redos, the workers' answers are printed as they are. The repository does not change.
- **A change** (`{"tasks": [...]}`): one `sandbox` run per task, on the branches `codetwine-loom/run/<run-id>/1`, `/2` and so on. A task that did not finish is handed out again on a new branch, up to `DISPATCH_MAX_RETRY` times. When all tasks have finished, `integrate` combines their branches. When the merge settles, the test-writing stage and the two refactoring stages run, each adding a commit to the same branch. The result up to the last settled stage is applied to the working tree. Nothing is committed to your branch.
- **Things to confirm** (`{"unclear": [...]}`): something only you can decide. `run` prints up to `DISPATCH_MAX_UNCLEAR` items, starts no agent and exits with code 1. Run it again with the answers added to the request.

The request is split, not the files: two tasks may touch the same file, and the merge resolves that. Every container starts from the working tree as it was when the run started.

**When even one task did not finish, nothing is merged.** The branches stay, and the exit code is non-zero. A task that ended with the tests red (`tests_red`) counts as finished; the merge's test check decides whether the union is worse than the base. When the request did not succeed, the last line gives the `--resume` names to pass, and with `DISPATCH_MAX_RERUN` at 1 or more `run` does that itself ([reruns](settings.md#dispatching-and-the-report-run)). When the working tree changed during the run and the union no longer applies, nothing is written; the output names the patch and the branch, and `integrate --resume <union>` applies it later.

### Progress lines

`run` prints one line to stderr at each break between parts of the run and for each finished task. Each line starts with the time since the run started:

```text
0:00:00 INFO codetwine_loom.progress: prepare the analysis: start
0:00:00 INFO codetwine_loom.progress: prepare the analysis: end (0s)
0:00:00 INFO codetwine_loom.progress: split the request: start
0:01:09 INFO codetwine_loom.progress: split the request: end (1m09s)
0:01:09 INFO codetwine_loom.progress: the request is 2 task(s); starting 2 agent(s) at once
0:01:09 INFO codetwine_loom.progress: run the tasks: start
[task 1/2] 0:05:01 INFO codetwine_loom.progress: task 1 of 2 finished (unverified) in 3m52s: 1 of 2 finished, 1 running
[task 2/2] 0:06:40 INFO codetwine_loom.progress: task 2 of 2 finished (unverified) in 5m31s: 2 of 2 finished, 0 running
0:06:40 INFO codetwine_loom.progress: run the tasks: end (5m31s)
0:06:40 INFO codetwine_loom.progress: combine the branches: start
```

| Line | Content |
|----|------|
| `<part>: start` / `end (<elapsed>)` / `stopped (<elapsed>)` | The start and end of each part: `prepare the analysis` (with `--doc-create`, also `make the design documents` or `make the design documents again`), `split the request`, `run the tasks`, `answer the questions`, `combine the branches`, `combine the answers`, `write the tests` (with `combine the test branches`), `refactor the change` / `refactor the tests`, and `write the report`. `stopped` means the part ended with an exception |
| `the request is <N> task(s); starting <P> agent(s) at once` | The number of tasks (`question(s)` for a question) |
| `the request leaves <N> thing(s) open for the person who asked; no agent is started` | The request came back with things to confirm |
| `tasks <n>, <m> change the same definition, so they run one after another` | Tasks that name the same definition run in order |
| `task <n> of <N> finished (<verdict>) in <elapsed>: <M> of <N> finished, <R> running` | An agent finished. A task that did not finish shows `did not finish` and is handed out again. Test writing says `test task` |

A prefix such as `[task 2/5] ` names the agent that printed the line (`[task 2/5 try 2] ` for a retry, `[question 1/3] `, `[test task 1/3] `). Warnings are printed too. Everything else goes to log files ([output.md](output.md#files)). With `AGENT_LOG_LEVEL=DEBUG`, every line is printed, including per-turn lines; with `AGENT_LOG_LEVEL=WARNING`, only warnings are. Questions from the tool appear only when both the input and the output are a terminal.

### The report at the end

At the end of a change, `run` prints an account of what was changed where, based on the diff applied to the working tree:

```text
(the lines of each worker's state and of the merge's result)

The request was to rename read_config to load_config. In codetwine_loom/read.py the
function is renamed to load_config, codetwine_loom/callers.py is a new file the callers
were moved to, and codetwine_loom/old.py, no longer used, was removed. tests/test_read.py
gains a test checking that the settings can be read under the new name.

changed:
  A codetwine_loom/callers.py
  A tests/test_read.py  (test)
  M codetwine_loom/read.py
  D codetwine_loom/old.py

to carry on from what this run left: run --resume ...
```

`REPORT_MODEL` writes the prose in the language of the request, from the request, the list of changed files and the diff. It also lists any cases the test-writing stage recorded as not done. `(test)` marks test files. Nothing is printed for a question, or when the union was not applied. When the prose cannot be written, or `REPORT_MODEL` is empty, only the file list appears.

## Carrying on from an earlier run (`run --resume`)

Branches remain when a run fails: those of workers that did not finish, of unsettled merges, and of stages that did not settle. Pass them with `--resume`, and the next run splits the request again, knowing where the previous run stopped.

```bash
# 1st round. Worker 2 used up its turns, and no merge was done
codetwine-loom run --project "$REPO" --output-dir ./out \
  --request "Add --version and --verbose to main.py"
```

```text
tasks: 2
  1. codetwine-loom/run/20260906-184500-1-1234/1 (7f18636a1b2c), unverified
  2. codetwine-loom/run/20260906-184500-1-1234/2 (2c9d41e3f5a0), failed
     it did not finish: exit status 1
combined: nothing (1 task(s) did not finish, so what the others left is on their branches and nowhere else)
tokens: 412,380 in 61 call(s): 398,204 input (341,120 read from the cache, 22,410 written to it), 14,176 output
record: /home/me/out/agents/.repo/codetwine/record/20260906-184500-1-1234.json
to carry on from what this run left: run --resume codetwine-loom/run/20260906-184500-1-1234/1 --resume codetwine-loom/run/20260906-184500-1-1234/2
```

```bash
# 2nd round. Pass the names the last line gave. The branches hold the request
codetwine-loom run --project "$REPO" --output-dir ./out \
  --resume codetwine-loom/run/20260906-184500-1-1234/1 \
  --resume codetwine-loom/run/20260906-184500-1-1234/2
```

`--request` can be omitted: each commit holds its run's original request, and that request is used when all the branches name the same one. A `--request` you pass is used instead, so you can resume with a narrower request such as "only the rest". For each branch, a block like this is put in front of the request for the splitting stage:

```markdown
## What an earlier run left on codetwine-loom/run/20260906-184500-1-1234/2

The repository you are reading does NOT hold this work. It is only on that branch, so
what is described here is not in the files you read.
- asked: Add a --verbose flag to parse_args() in main.py and …
  did: Added --verbose flag and reordered main() …
  ended: failed - the turns ran out before it said the change was done
```

All of this is read from the commits, so you can carry on from another machine as long as the branches exist. Passing the one merge branch is enough, since its commit has every worker's commit as a parent. A branch that does not exist is refused before any container starts.

The branch is only read, never checked out. The workers start from the current working tree, so to build on the previous unfinished code, apply that branch's diff to the working tree first (or combine partial results with `integrate`). Branches outside `codetwine-loom/run/` can also be passed, but the cleanup of old runs does not remove them.

## Exit codes and how a run ends (`COMPLETE()`)

Every stage ends when the model calls `COMPLETE()`. Workers that answer questions and the merge end with `COMPLETE("<answer>")`. When a request needs no change, the change stage ends with `COMPLETE("<why-not-needed>")`, which is recorded as `no_change`; the first such `COMPLETE()` is sent back once (`PLAN_COMPLETE_GUARD`). A `COMPLETE()` that breaks a rule (too long, too many things to confirm, a split in the wrong form) is sent back once; if it is still too long, it is cut, and if the form is still wrong, the run stops.

| Exit code | When |
|------|------|
| `0` | The verdict is `unverified`, `no_change` or `tests_red`, and for `sandbox`, `run` and `integrate` the result was applied to the working tree. A question run that produced a combined answer, even if a worker failed |
| Non-zero (usually `1`) | The verdict is `failed`: the run stopped part-way, wrote nothing, used up its turns without `COMPLETE()`, or left the program not working. Also: `run` returned things to confirm, a task did not finish, the result was not applied to the working tree, the test-writing stage recorded cases the change does not do (the result is still applied), or `MAX_TOKEN_PER_RUN` was reached |
| `130`, `143`, `129` | Stopped by Ctrl-C, `SIGTERM` or `SIGHUP`. stderr shows `stopped by <signal>` and the run is recorded as `failed` |

`tests_red` means the run finished with the tests red; `run` does not hand such a task out again but takes it to the merge. When a run uses up `RLM_MAX_TURN` without `COMPLETE()`, `EDIT_LATE_ANSWER` (on by default) asks once what it changed and records the reply; the verdict stays `failed`. When a run of `run`, `sandbox` or `integrate` leaves something to carry on from, its last line gives the command.

## Output of `history`, `build` and `search`

`history` prints two lists. `runs:` lists the records of `run`, `sandbox` and `integrate`, oldest first, with the name, subcommand, exit code, time, tokens, request, and the command to carry on. Records whose result the working tree already holds are not shown. A run that was not applied because of the paths it touches shows the reason and the command to apply it by hand. `history:` has one line per `edit` change, where `add` / `mod` / `del` count the files added, modified and deleted:

```text
runs: /home/me/out/agents/.repo/codetwine/record
  20260906-184500-1-1234  run  exit 1  6m12s  412,380 token(s)
    Add --version and --verbose to main.py
    to carry on from it: run --resume codetwine-loom/run/20260906-184500-1-1234/1 --resume codetwine-loom/run/20260906-184500-1-1234/2
history: ./out/codetwine/edit_history.json
  1  2026-08-23T17:34:26  unverified (confirmed: -)  add1 mod2 del0
    nothing here checks that the request itself is done, so the work may still be part-way
```

`build` prints whether it reused the analysis (`reused existing`) or ran codetwine (`ran codetwine`), the numbers of definitions, files and calls, the sizes of the definition and file indexes, the primary language and the index's path:

```text
knowledge: ./out/codetwine/project_knowledge.sqlite (reused existing)
defs: 203 files: 19 calls: 88
terms: 2538 avg_len: 245.5
file terms: 616 file avg_len: 99.5
primary_ext: py
index: ./out/codetwine/code_index.sqlite (232.0 KB)
```

`search` prints:

```text
query: config setting load env value parse
confidence: high (raw 0.9 / hit_terms 5 / spread 7)
files:
  codetwine/config/settings.py
  codetwine/llm/client.py
defs:
  codetwine/config/settings.py::get_config_value  [412-478]
    def get_config_value(key, default=None):
  codetwine/llm/client.py::generate  [96-112]
        async def generate(
expanded:
  main.py::main  [10-40]  <- codetwine/config/settings.py::get_config_value
file_summaries:
  codetwine/config/settings.py
    `settings.py` is CodeTwine's central config module: loads env vars ...
callers:
  codetwine/llm/client.py::LLMClient
    <- main.py::main
```

`query` is the search terms from step 1 (the request itself when step 1 was skipped). Each definition is shown with its line range and first line, and each file with its summary. `expanded` lists definitions added by following call relations, with the definition they were reached from after `<-`. `spread` is the number of files the hit definitions span; a larger number means a more cross-cutting change. When nothing is hit, or step 1 failed, a line under `query` says so.
