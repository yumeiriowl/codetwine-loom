---
name: codetwine-loom
description: Set up and drive the codetwine-loom command-line tool, which changes code in a git repository from a request in plain language, or answers questions about the code, using LLM agents in containers. Covers installing it, writing its .env, building its container image, choosing a subcommand (build, search, run, sandbox, answer, edit, integrate, history and the others), reading its output and exit codes, carrying on from a failed run, and writing the instruction files (CODETWINE_LOOM.md and the others) the target repository can hold. Use this skill whenever the user asks you to use, run, install, configure or set up codetwine-loom, to hand a change or a question about a repository to it, to carry on from or undo one of its runs, or to write instructions for it, even if they only name the tool in passing.
license: MIT
compatibility: Requires git, Python 3.11 or later, uv 0.11.8 or later, Podman or Docker, a shell, and network access to an LLM provider
---

# codetwine-loom — setting it up and using it

codetwine-loom takes a request in plain language and a git repository (the *target repository*). It builds a search index from a codetwine analysis of the repository, finds the files the request is about, and has LLM agents answer the question or change the code and run the tests. Changes are made in a container with no network and no API key, and a successful result is applied to the working tree of the target repository **without committing**.

You run it from a shell as the `codetwine-loom` command. Run `codetwine-loom <subcommand> --help` when you need an argument this skill does not list.

Detailed references in this skill:

- [references/commands.md](references/commands.md) — every subcommand, every argument, the output lines and the exit codes
- [references/settings.md](references/settings.md) — the settings (environment variables) worth knowing, grouped by purpose
- [references/instruction-files.md](references/instruction-files.md) — `CODETWINE_LOOM.md` and the other instruction files, with templates

## 1. Ground rules

- **Use `run` or `sandbox` for changes.** `edit` and `answer` run on the host, and the Python the model writes runs with your permissions there. Use `edit` only when the user explicitly asks for it and accepts that risk.
- **Never commit for the user.** A run applies its change to the working tree; `HEAD` and the branch do not move. Show the user the result (`git -C <repo> diff --stat`, then the parts that matter) and commit only when they ask.
- **Exit status 0 does not mean the request is done.** It means the result was applied. The tests may be red (`tests_red`), and nothing checks that the change meets the request. Read the diff before you tell the user it is finished.
- **One command at a time per working tree.** Do not start a second `codetwine-loom` command on the same target repository while one runs, and do not edit the target's working tree during a run (the result may then not apply). Parallel work is `run`'s job.
- **Keep the tool's files out of the target repository.** Run the command from a directory outside the target repository, keep `.env` and the output directory (`--output-dir`) there, and write request files there too.
- **Start it with the `codetwine-loom` command** (or `<clone>/.venv/bin/codetwine-loom`), never with `python -m codetwine_loom` from inside the target repository.
- **API keys**: do not ask the user to paste a key into the conversation, and never print `.env`. Ask them to write the key into `.env` themselves, or to export it in the shell that runs the command.

## 2. Non-interactive behaviour

The tool asks questions only when both its input and its output are a terminal. A shell an agent drives is normally not a terminal, so:

| Situation | At a terminal | When you run it (no terminal) |
|---|---|---|
| No `.env` and a key is missing | A guide asks for the settings and writes `.env` | Stops, naming the missing variable. Write `.env` (section 4) |
| `codetwine-loom setup` | Asks every setting, writes `.env`, checks the image | Refuses to run. Ask the user to run it in their own terminal, or write `.env` yourself |
| The container image is missing | Asks `Build it now? [Y/n]` | Stops and prints the build command. Run that command (section 5) |
| The container image is stale | Asks `Build it again now? [Y/n]` | Warns, prints the build command, and goes on with the old image |
| `run` with no analysis yet | Asks whether to write design documents | Makes the analysis without design documents |
| `edit` after an unfinished `edit` | Asks whether to carry on | Starts fresh (pass `--resume` to carry on) |
| COBOL files and no `COBOL_DIALECT` | `run` / `integrate` ask once | Not asked; set `COBOL_DIALECT` yourself when the code needs a dialect |

## 3. Installing

Check first: `command -v codetwine-loom`, or a clone of the codetwine-loom repository (it holds `pyproject.toml`, `uv.lock` and `codetwine_loom/`). If neither exists, clone it outside the target repository: `git clone https://github.com/yumeiriowl/codetwine-loom.git`.

Prerequisites: `git`, Python 3.11 or later, [uv](https://docs.astral.sh/uv/) 0.11.8 or later, and Podman or Docker (for `run`, `sandbox`, `integrate` and `warm`). The target repository must be a git repository, and its directory must not be named `agents`.

```bash
cd <codetwine-loom-clone>
uv sync --locked                       # installs into .venv at the locked versions
.venv/bin/codetwine-loom --help        # or: source .venv/bin/activate
```

- An older uv fails with `Failed to parse` or `Required uv version`: run `uv self update`.
- Keep the target repository outside the codetwine-loom clone.
- To move codetwine (the analyser) to its latest commit: `uv sync --upgrade-package codetwine`. The image is then stale and should be rebuilt.

## 4. Configuring (`.env`)

Every setting is an environment variable. A value comes from the first of: a command-line argument, the environment, `.env`, the `PRESET` column, the default. **Only the `.env` in the current directory is read**, once at start-up; parent directories are not searched. A `.env` that git tracks is not read, and if it holds this tool's settings every subcommand stops.

Write `.env` in the directory you will run the command from (usually the codetwine-loom clone):

```bash
cd <codetwine-loom-clone>
cp codetwine_loom/.env.example .env    # then the user writes LLM_API_KEY into it
chmod 600 .env
```

Normally only these are needed:

| Variable | Meaning | Default |
|---|---|---|
| `LLM_API_KEY` | The API key for the models. It serves one provider | empty |
| `PROJECT_DIR` | The target repository (`--project` gives it too) | none; required |
| `PRESET` | `low` / `medium` / `high`: accuracy, cost and time together | `medium` |
| `MEDIUM_MODEL` | Model of the stages that do the work (litellm form, `provider/model`) | `anthropic/claude-sonnet-5-5` |
| `HIGH_MODEL` | Model of the stages that decide what to do (planning, splitting, merging) | `anthropic/claude-opus-5-5` |

- Models of **two or more providers**: leave `LLM_API_KEY` empty and set each provider's own variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
- Ask the user before changing the models or `PRESET`; they decide the cost.
- At start-up, every subcommand that calls a model checks that each model has its key, and stops naming the missing variable. A value a setting cannot take stops the command with one line naming the setting.
- Behind a proxy, set `HTTPS_PROXY` / `NO_PROXY`, and `SSL_CERT_FILE` for a proxy with its own certificate authority.

Other settings (tests, cost limits, containers, web search, logging) are in [references/settings.md](references/settings.md). Pass a one-off value on the command line instead of editing `.env`: `TEST_COMMAND="uv run --with pytest pytest -q" codetwine-loom run ...`.

## 5. The container image

`run`, `sandbox`, `integrate` and `warm` run containers from the image `codetwine-loom`, which holds the toolchains of every language. Without a terminal, a missing image stops the command with a message like:

```text
Build it (TOOLCHAIN=c cbl cs java js kt r rs ts) with:
  podman build ... --label io.codetwine-loom.source=<hash> ... -t codetwine-loom <clone>
```

Run that printed command as it is (tell the user first: it fetches packages over the network, uses several GB of disk, and takes a few minutes, longer on a slow network, so run it in the background or with a long timeout). Do not drop the `--label`, or the image counts as stale. A stale image (after upgrading codetwine-loom or codetwine) is used with a warning that prints the same kind of command; rebuild with it before a run whose result matters.

- The engine is `podman`, then `docker`. Force one with `CONTAINER_ENGINE=docker` (or `--engine docker`).
- To build a smaller image with some languages only: `podman build --build-arg TOOLCHAIN="java ts" -t codetwine-loom-java-ts <clone>` and pass `--image codetwine-loom-java-ts`. Languages it lacks cannot be tested in it. An image built this way lacks the tool's source label, so runs report it as stale; the rebuild command they print keeps the same `TOOLCHAIN`, and running it clears the warning.

## 6. Choosing the subcommand

| The user wants | Use |
|---|---|
| A change of any size | `run --request "..."` (splits it into tasks, runs them in parallel, merges, writes tests, refactors, applies) |
| One small, focused change, faster and cheaper | `sandbox --request "..."` |
| An answer about the code | `run --request "..."` (splits it into sub-questions and combines the answers; changes nothing). `answer --request "..."` is quicker but runs on the host with the same risk as `edit`; use it only when the user accepts that |
| Only the files and definitions a request is about | `search --request "..."` (changes nothing) |
| To combine existing branches | `integrate --branch A --branch B` (merged in that order) |
| To carry on from a run that left something | The command the run's `to carry on from what this run left:` line printed (section 8) |
| To see earlier runs | `history` |
| The analysis and index only, or design documents | `build` / `build --doc-create` |

Every command needs `--project <target-repo>` (or `PROJECT_DIR`) and uses `--output-dir` (default `./out`, relative to the current directory). The output directory must not be inside the target repository.

## 7. Running a request

### Before the first run

```bash
cd <directory-with-.env>
REPO=/abs/path/to/target-repo
codetwine-loom build --project "$REPO" --output-dir ./out
```

`build` makes the codetwine analysis and the search index, without calling an LLM. It is optional (the other commands make a missing analysis themselves) but shows early whether the setup works. Its output names the primary language (`primary_ext:`).

`build --doc-create` also has codetwine write design documents for each file, which improves search but **calls the LLM several times per file**. Ask the user before using it on a large repository. Later automatic rebuilds do not write design documents; run `build --doc-create` again to update them.

### Writing the request

Write the request as you would brief a developer: what should change, where if you know, and what must stay as it is. Any language works. For a multi-line request, use `--request-file` with a file outside the target repository, or standard input:

```bash
codetwine-loom run --project "$REPO" --output-dir ./out --request-file - <<'EOF'
Make loading the settings file report a clear error instead of crashing when the file is missing.
Keep the public function names as they are.
EOF
```

A request that is itself one of the tool's flags (such as `--dry-run`) must be passed as `--request=<text>`.

### Running it

Runs take from a few minutes to over an hour, and `run` prints progress lines to stderr (`<part>: start`, `task 1 of 3 finished ...`). Run the command in the background or with a long timeout, and wait for it to end; do not start anything else on the same target repository meanwhile.

```bash
codetwine-loom run --project "$REPO" --output-dir ./out --request "..."
```

Useful flags (all in [references/commands.md](references/commands.md)):

- `--preset low|medium|high` for this run only
- `--no-test-create --no-refactor --no-test-refactor` (`run`, `integrate`) to skip the stages after the merge
- `--no-overview` to skip the overall overview (one fewer LLM step at the start)
- `--dry-run` (`sandbox`, `edit`) to write nothing
- `--guide <path>` (`sandbox`, `edit`) for another instruction file
- `--exclude NAME ...` (`build`, `run`, `sandbox`, `integrate`) to keep directories out of the analysis

Cost limits are off by default. When the user gives a budget, set `MAX_TOKEN_PER_RUN` (tokens per run of `run`, `sandbox` or `integrate`) or `LLM_MAX_CALL` (model calls per agent).

### What goes into the container

The container gets the working tree as it is, including uncommitted changes and untracked files, **but not files `.gitignore` ignores and not submodule contents**. A change that touches an untracked `.env`, an ignored path, a nested repository, git's configuration or hooks is not applied at all.

## 8. Reading the result

### The last lines

A successful `sandbox`:

```text
base: 3f9c1a2b...
patch: .../sandbox/patch-....diff (2.1 KB, 3 file(s))
applied to the working tree: 3 file(s), over 3f9c1a2b4d5e. Commit them yourself
tokens: 96,412 in 14 call(s): ...
record: .../record/....json
to take it back out of the working tree: git -C <repo> apply --binary --whitespace=nowarn --no-ignore-whitespace -R <out>/agents/.repo/<repo>/land-last.diff
```

A change by `run` ends with a short report of what changed where (written in the language of the request) and a `changed:` list (`A` added, `M` modified, `D` deleted, `(test)` for test files). A question ends with the combined answer.

| Line | What to do |
|---|---|
| `applied to the working tree: ...` | Review `git -C "$REPO" diff` and summarise it for the user |
| `to take it back out of the working tree: <command>` | Keep the command. Run it only when the user wants the change removed |
| `to carry on from what this run left: <command>` | Something is left to carry on from: the run did not succeed, or (with exit 0) a stage after the merge did not settle and the result was applied only up to the last settled stage. Offer to run the printed command (section "Carrying on and undoing") |
| `NOT applied to the working tree: <reason>` | Nothing was written. Tell the user the reason; the work is on the named branch |
| `NOT done by the change, as the tests found (<N>):` | The written tests found cases the change does not do. The exit code is 1, but the change **is** in the working tree: review the diff and report the cases |
| `tests: none written (...)`, `refactoring: nothing done (...)` | The stage was skipped for the reason given; not an error |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Applied (or answered). The verdict is `unverified`, `no_change` or `tests_red` |
| `1` (non-zero) | `failed`, things to confirm (below), a task did not finish, the result was not applied, the written tests found cases the change does not do (the change is still applied), or `MAX_TOKEN_PER_RUN` was reached (changes applied before that stay). Check `git -C "$REPO" status` rather than assuming nothing changed |
| `130` / `143` / `129` | Stopped by Ctrl-C / `SIGTERM` / `SIGHUP` |

The verdict is never `passed`: `unverified` means nothing checked that the request is met. `tests_red` means the run finished with tests failing. `no_change` means the model decided no change was needed and said why.

### Things to confirm

When the request leaves a decision only the user can make, `run` prints up to 3 questions, starts no agent and exits with 1. Ask the user those questions, then run again with the answers added to the request. Do not answer them yourself unless the user told you to decide.

### Carrying on and undoing

- **Carry on**: the `to carry on from what this run left:` line prints a subcommand and its arguments, such as `run --resume codetwine-loom/run/<run-id>/1 --resume codetwine-loom/run/<run-id>/2` or `integrate --resume codetwine-loom/sandbox/<run-id>`. Run it as printed, with `codetwine-loom` in front and `--project "$REPO" --output-dir ./out` added. When it ends with `--request <request>`, replace `<request>` with the original request (quoted). Otherwise `--request` can be omitted (the branches hold it) or given to narrow the work ("only the rest"). `history` lists earlier runs with the command to carry on from each.
- **A second round** on top of an applied result: just run the next request. The base copies the working tree, committed or not. A container run does not remember the previous one; use `--resume` to tell it.
- **Undo**: run the `to take it back out of the working tree:` command. It works only while the applied lines are unchanged.
- **Read a failed run's work** without applying it: `git -C "$REPO" diff --no-textconv --no-ext-diff <base> codetwine-loom/sandbox/<run-id>` (`<base>` is the run's `base:` line; the two flags keep the user's diff drivers from running on model-written content). Applying it by hand skips the tool's safety checks, so do it only when the user asks.
- The tool keeps branches under `codetwine-loom/run/` and `codetwine-loom/sandbox/` for resumable runs and removes old ones itself. Do not delete them unless the user asks.

## 9. Instruction files for the target repository

By default, how to make a change is left to the model. The repository's conventions reach it only through the request or an instruction file at the top of the target repository:

| File | Read by |
|---|---|
| `CODETWINE_LOOM.md` | The planning and change stages (`edit`, `sandbox`, `run`'s workers) |
| `CODETWINE_LOOM_TEST.md` | The stage that writes tests after the merge |
| `CODETWINE_LOOM_REFACTOR.md` | Implementation refactoring after the merge |
| `CODETWINE_LOOM_TEST_REFACTOR.md` | Test refactoring after the merge |

Each stage reads only its own file; all are optional. `AGENTS.md` and `CLAUDE.md` are not read (set `EDIT_GUIDE=AGENTS.md` to use one). For container runs the file must be inside the repository and not ignored by git. The tool's own instructions win where they disagree.

When the user wants the tool to follow the repository's conventions every time, offer to write these files (they appear in `git status` as new files). How to write them, with templates, is in [references/instruction-files.md](references/instruction-files.md).

## 10. When something stops

| Message or symptom | Fix |
|---|---|
| A model setting lacks a variable | Have the user put the key in `.env`; with two providers, use the providers' own variables |
| `podman` / `docker` not found, or the engine does not answer | Install one, or set `CONTAINER_ENGINE` to the one that works |
| The image is missing | Run the printed build command (section 5) |
| `OUTPUT_DIR` is inside `PROJECT_DIR`, or holds another repository with the same name | Use another `--output-dir` outside the target repository |
| A tracked `.env` holds this tool's settings | Run the command from a directory whose `.env` git does not track |
| `sandbox` / `run` stop with different `--exclude` names | Run the `build --exclude ...` command they print |
| `run_test()` keeps saying `did not run`, or tests are not found | Set `TEST_COMMAND` (and `TEST_PATH`); see [references/settings.md](references/settings.md) |
| Exit status 137 | The container hit `SANDBOX_MEMORY` (default `2g`); raise it |
| A change that adds dependencies fails its tests | Refills of the package cache are off (`WARM_MAX_REFILL=0`), used up, or failed (no network or proxy). Raise `WARM_MAX_REFILL` only for requests the user trusts |
| The target repository has a branch named `codetwine-loom`, `codetwine-loom/run` or `codetwine-loom/sandbox` | Ask the user to rename it; the container commands refuse to start |

For anything not covered, read `codetwine-loom <subcommand> --help` and the tool's `README.md` and `docs/`.
