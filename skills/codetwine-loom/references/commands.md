# Subcommands, arguments and output

## Contents

- [Subcommands](#subcommands)
- [Arguments](#arguments)
- [Examples](#examples)
- [How `run` handles a request](#how-run-handles-a-request)
- [Progress lines](#progress-lines)
- [Output of `history`, `build` and `search`](#output-of-history-build-and-search)
- [Exit codes and verdicts](#exit-codes-and-verdicts)
- [Where the files go](#where-the-files-go)

## Subcommands

| Subcommand | What it does | Calls an LLM | Container |
|---|---|---|---|
| `build` | Makes the codetwine analysis and the search index. `--doc-create` also has codetwine write design documents | Only with `--doc-create` | No |
| `search` | Shows the candidate files and definitions for a request. Changes nothing | Yes (turns the request into search terms) | No |
| `run` | Reads the request in a container. A question is split into sub-questions whose answers are combined. A change is split into tasks run in parallel; the results are merged, given tests, refactored and applied to the working tree | Yes | Yes |
| `answer` | Answers one question on the host. Changes nothing | Yes | No |
| `edit` | Makes a change directly on the host | Yes | No |
| `sandbox` | Makes one change in a container and applies the result to the working tree | Yes | Yes |
| `integrate` | Combines two or more branches into one in a container, then runs the stages after the merge, and applies the result | Yes (the merge itself only when the machine checks leave something) | Yes |
| `test-create`, `refactor`, `test-refactor` | The stages after a merge. `run` and `integrate` run them inside the container; you do not normally call them | Yes | — |
| `overview` | Prepares the analysis and the index, then makes only the overall overview | Yes | No |
| `warm` | Fetches the test dependencies into the package cache, in a container with a network and no keys | No | Yes |
| `run-tests` | Runs the test commands once | No | No |
| `broker` | Makes the model calls of containers on the host. `sandbox`, `run` and `integrate` start it themselves | Yes | No |
| `format` | Runs `FORMAT_COMMAND` on the files changed since `HEAD` | No | No |
| `history` | Lists earlier runs of `run`, `sandbox` and `integrate` (with the command to carry on) and the changes `edit` made | No | No |
| `setup` | Asks for the settings, writes `.env` and checks the image. Terminal only | — | — |

`search`, `run`, `answer`, `edit` and `sandbox` first make the overall overview (`overview.json`, a short description of the repository's purpose and features used by search) when it is missing or stale; `--no-overview` skips it.

`search`, `answer`, `edit` and `run-tests` make a missing analysis without design documents and carry on. A command that would rewrite the analysis stops while a `sandbox`, `run` or `integrate` run is using the same output directory.

## Arguments

| Argument | Subcommands | Description | Default |
|---|---|---|---|
| `--project PATH` | all except `setup` | The target repository (`.` for the current directory). Required except for `broker` | `PROJECT_DIR` |
| `--output-dir PATH` | all except `setup` | Where the analysis, the index and the run records go. Must be outside the target repository | `OUTPUT_DIR` (`./out`) |
| `--request TEXT` | `search`, `run`, `answer`, `edit`, `sandbox`, `integrate` | The request. `run` decides whether it is a question or a change. Required for `search`, `answer`, `edit`, `sandbox`, and `run` without `--resume`. Write a request that is itself a flag of the tool as `--request=<text>`. Up to 131,008 bytes for the container commands | — |
| `--request-file PATH` | same as `--request` | Reads the request from a UTF-8 file; `-` reads standard input. Not with `--request` | — |
| `--doc-create` | `build` | Always runs codetwine and has it write design documents for files without one, changed files and files that depend on them. Several LLM calls per file | Off |
| `--doc-create` | `sandbox`, `run` | Writes design documents on the host once at the start, as `build --doc-create` does | Off |
| `--doc-create` | `integrate` | Writes design documents when there is no analysis yet | Off |
| `--exclude NAME ...` | `build`, `sandbox`, `run`, `integrate` | Directory and file names kept out of the analysis. A name without `/` matches at any depth; with `/` it is a path from the repository root and may use `fnmatch` patterns (`./vendor`, `apps/*/gen`). Recorded with the analysis; the container commands stop when given different names and print the `build --exclude ...` to run | None |
| `--no-overview` | `search`, `run`, `answer`, `edit`, `sandbox` | Does not make or use the overall overview | Made and used |
| `--preset low\|medium\|high` | `run`, `answer`, `edit`, `sandbox`, `integrate` | The preset for this run | `PRESET` (`medium`) |
| `--dry-run` / `--no-dry-run` | `edit`, `sandbox` | The change tools check but write nothing. `edit --dry-run` prints the diff it would have written; `sandbox --dry-run` puts nothing on a branch or in the working tree | Writes |
| `--check` / `--no-check` | `edit`, `sandbox` | Holds the change to the static check (`CHECK_COMMAND`) and the references the analysis resolves; findings left at the end make the verdict `failed` | Checked |
| `--guide PATH` | `edit`, `sandbox` | The instruction file for changes, relative to the repository. `""` reads none. `run` has no such argument; set `EDIT_GUIDE` | `EDIT_GUIDE` (`CODETWINE_LOOM.md`) |
| `--resume` / `--no-resume` | `edit` | Carry on from the previous unfinished `edit`, or start fresh. Without either, `edit` asks at a terminal and otherwise starts fresh | Asks |
| `--resume NAME` | `run`, `integrate` | Carry on from a branch an earlier run left. `run` takes it several times; `integrate` takes one union branch | — |
| `--branch NAME` | `integrate` | A branch to combine. Give two or more; merged in that order. Not needed with `--resume` | — |
| `--no-test-create`, `--no-refactor`, `--no-test-refactor` | `run`, `integrate` | Skip the test-writing, implementation refactoring or test refactoring stage after the merge | All run |
| `--engine NAME` | `sandbox`, `run`, `integrate` | The container engine | `CONTAINER_ENGINE` (`podman`, then `docker`) |
| `--image NAME` | `sandbox`, `run`, `integrate` | The container image | `CONTAINER_IMAGE` (`codetwine-loom`) |
| `--endpoint URL` | `broker` | `unix://<path>` or `tcp://<host>:<port>`. A `tcp://` broker has no authentication | `LLM_BROKER` |

`overview`, `warm`, `run-tests`, `format` and `history` take only `--project` and `--output-dir`. `setup` takes none.

Arguments such as `--here`, `--tasks-out`, `--answered`, `--answer-out`, `--onto`, `--onto-commit`, `--outcome-out`, `--gap-in`, `--base`, `--target-in`, `--target-only`, `--agent`, `--run`, `--fill-ext`, `--token-out`, `--token-left` and `--label` are passed by the tool to its own containers and broker. Do not pass them.

## Examples

```bash
REPO=/abs/path/to/target-repo

# The analysis and index (no LLM); with design documents (calls the LLM per file)
codetwine-loom build --project "$REPO" --output-dir ./out
codetwine-loom build --project "$REPO" --output-dir ./out --doc-create
codetwine-loom build --project "$REPO" --output-dir ./out --exclude vendor "apps/*/gen"

# Candidates for a request (changes nothing)
codetwine-loom search --project "$REPO" --output-dir ./out --request "Keep loading the settings from crashing"

# A question, split and combined (changes nothing)
codetwine-loom run --project "$REPO" --output-dir ./out \
  --request "Where are the settings loaded? What happens when the environment variable is missing?"

# A change, split, run in parallel, merged, tested and refactored
codetwine-loom run --project "$REPO" --output-dir ./out \
  --request "Tidy up how the settings are loaded, and update the callers to match"

# The same without the overview and the stages after the merge
codetwine-loom run --project "$REPO" --output-dir ./out --request "..." \
  --no-overview --no-test-create --no-refactor --no-test-refactor

# One change in a container
codetwine-loom sandbox --project "$REPO" --output-dir ./out --request "Keep loading the settings from crashing"

# Carry on from a run that failed (names from its last line)
codetwine-loom run --project "$REPO" --output-dir ./out \
  --resume codetwine-loom/run/20260906-184500-1-1234/1 --resume codetwine-loom/run/20260906-184500-1-1234/2

# Combine branches
codetwine-loom integrate --project "$REPO" --output-dir ./out --branch feature/a --branch feature/b

# Fill the package cache by hand; list earlier runs
codetwine-loom warm --project "$REPO" --output-dir ./out
codetwine-loom history --project "$REPO" --output-dir ./out
```

## How `run` handles a request

A read-only stage in a container first splits the request. It returns one of three results:

- **A question**: one worker per sub-question answers it (up to `DISPATCH_PARALLEL` at once, reading only). A merge run combines the answers. The repository does not change.
- **A change**: one `sandbox` run per task, on branches `codetwine-loom/run/<run-id>/1`, `/2`, …. A task that did not finish is handed out again (up to `DISPATCH_MAX_RETRY`). When every task has finished, `integrate` merges the branches; when the merge settles, the test-writing stage and the two refactoring stages each add a commit. The result up to the last settled stage is applied to the working tree.
- **Things to confirm**: decisions only the user can make. `run` prints them, starts no agent and exits with 1.

When even one task did not finish, nothing is merged; the branches stay and the exit code is non-zero. A request that did not succeed is run again from its branches up to `DISPATCH_MAX_RERUN` times (`runs: N` at the top of the output). Every container starts from the working tree as it was when the run started.

## Progress lines

`run` prints a line to stderr at each break, each starting with the elapsed time:

```text
0:00:00 INFO codetwine_loom.progress: split the request: start
0:01:09 INFO codetwine_loom.progress: the request is 2 task(s); starting 2 agent(s) at once
[task 1/2] 0:05:01 INFO codetwine_loom.progress: task 1 of 2 finished (unverified) in 3m52s: 1 of 2 finished, 1 running
0:06:40 INFO codetwine_loom.progress: combine the branches: start
```

The parts are `prepare the analysis`, `split the request`, `run the tasks` / `answer the questions`, `combine the branches` / `combine the answers`, `write the tests`, `refactor the change`, `refactor the tests` and `write the report`. `stopped` after a part means it ended with an exception. `AGENT_LOG_LEVEL=DEBUG` prints every line, including per-turn lines; `WARNING` prints only warnings.

## Output of `history`, `build` and `search`

`history`:

```text
runs: /home/me/out/agents/.repo/myrepo/record
  20260906-184500-1-1234  run  exit 1  6m12s  412,380 token(s)
    Add --version and --verbose to main.py
    to carry on from it: run --resume codetwine-loom/run/20260906-184500-1-1234/1 --resume codetwine-loom/run/20260906-184500-1-1234/2
history: ./out/myrepo/edit_history.json
  1  2026-08-23T17:34:26  unverified (confirmed: -)  add1 mod2 del0
```

Runs whose result the working tree already holds are not listed.

`build` prints whether it reused the analysis or ran codetwine, the counts of definitions, files and calls, the primary language (`primary_ext:`) and the index path.

`search` prints `query:` (the search terms), `confidence:` (`high` or `low`), `files:`, `defs:` (with line ranges and first lines), `expanded:` (definitions reached through call relations), `file_summaries:` and `callers:`.

## Exit codes and verdicts

| Exit code | When |
|---|---|
| `0` | The verdict is `unverified`, `no_change` or `tests_red`, and for `sandbox`, `run` and `integrate` the result was applied. A question that produced a combined answer |
| Non-zero (usually `1`) | The verdict is `failed` (stopped part-way, wrote nothing, used up its turns, or left the program not working); `run` returned things to confirm; a task did not finish; the result was not applied; the written tests found cases the change does not do (the result is still applied); `MAX_TOKEN_PER_RUN` was reached |
| `130`, `143`, `129` | Stopped by Ctrl-C, `SIGTERM` or `SIGHUP`; recorded as `failed` |

| Verdict | Meaning |
|---|---|
| `unverified` | The run finished; nothing checked that the request is met |
| `tests_red` | The run finished with tests failing |
| `no_change` | The model decided no change was needed and said why |
| `failed` | See exit code 1 |
| `settled` / `unsettled` | For a merge and the stages after it: whether the machine checks pass |

A failed `sandbox` leaves its work on `codetwine-loom/sandbox/<run-id>`; an unsettled merge leaves the union there too.

## Where the files go

```text
<output-dir>/
  <repo-name>/                     made by build
    project_knowledge.sqlite       codetwine's analysis
    code_index.sqlite              the search index
    overview.json                  the overall overview
    edit_history.json              records of edit on the host
    turns/                         logs of runs on the host
  agents/
    .repo/<repo-name>/
      record/<run-id>.json         the record of one run (history reads it)
      land-last.diff               the diff the last command applied to the working tree
      cache/                       the package cache shared by containers
    <agent>/<repo-name>/turns/     that agent's log (.log) and turns (.jsonl)
```

`<repo-name>` is the directory name of the target repository. One output directory can serve several target repositories with different directory names.
