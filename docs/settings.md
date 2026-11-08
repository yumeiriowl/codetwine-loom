# Settings

This page lists every setting of codetwine-loom. The README's [Settings](../README.md#️-settings) shows only the minimal settings and the `PRESET` table.

## Contents

- [How the values are read](#how-the-values-are-read)
- [The minimal settings and `PRESET`](#the-minimal-settings-and-preset)
- [Checking the keys and the image at start-up](#checking-the-keys-and-the-image-at-start-up)
- [Proxies and certificates](#proxies-and-certificates)
- [Paths and analysis](#paths-and-analysis)
- [Index building and search](#index-building-and-search)
- [The overall overview](#the-overall-overview)
- [Models and model calls](#models-and-model-calls)
- [Agents](#agents)
- [Dispatching and the report (`run`)](#dispatching-and-the-report-run)
- [Merging (`integrate`)](#merging-integrate)
- [Edit history](#edit-history)
- [Writing tests after the merge (`TEST_CREATE`)](#writing-tests-after-the-merge-test_create)
- [Refactoring after the merge (`REFACTOR` / `TEST_REFACTOR`)](#refactoring-after-the-merge-refactor--test_refactor)
- [The instruction files for changes (`CODETWINE_LOOM.md`)](#the-instruction-files-for-changes-codetwine_loommd)
- [The planning stage (two stages)](#the-planning-stage-two-stages)
- [The explorer](#the-explorer)
- [Reading tools (`read_source_file`, `grep_source`)](#reading-tools-read_source_file-grep_source)
- [Asking about facts outside the repository (`web_search`)](#asking-about-facts-outside-the-repository-web_search)
- [Reading a page of the search results (`web_fetch`)](#reading-a-page-of-the-search-results-web_fetch)
- [Change tools](#change-tools)
- [Tests and the static check](#tests-and-the-static-check)
- [Running the program (`run_project`)](#running-the-program-run_project)
- [Automatic check after each edit (`SMOKE_ARG`)](#automatic-check-after-each-edit-smoke_arg)
- [Sandbox](#sandbox)
  - [Repositories whose tests need another service (`SANDBOX_NETWORK`)](#repositories-whose-tests-need-another-service-sandbox_network)
- [The broker and the logs](#the-broker-and-the-logs)
- [Defaults by language](#defaults-by-language)

## How the values are read

Every setting is an environment variable with exactly the name in the tables below, with no prefix. A value is taken from the first of these that gives one: a command-line argument, the environment, `.env`, the `PRESET` column, the default. A default written as "the value of `X`" follows `X` when the setting is not set.

Only one `.env` is read: the one in the directory the command is run in, once at start-up. Parent directories are not searched. It fills the names the shell leaves unset or empty. The template is `codetwine_loom/.env.example`; it holds the minimal settings and the ones most often changed. When there is no `.env`, the [first-run guide](usage.md#the-first-run-guide-and-setup) can write one. Containers never read a `.env`; they get the settings by name. `PYTHON_DOTENV_DISABLED=1` turns off reading, asking for and writing `.env`.

Keep this tool's `.env` apart from the target repository. **A `.env` that git tracks is not read.** When such a file sets any setting of this tool, every subcommand stops before starting and names the file and the settings ([details](limitations.md#a-env-the-target-repository-tracks-is-not-read)). A tracked `.env` that holds none of this tool's settings is skipped silently.

An empty value (`NAME=`) means the default, except for settings whose default is empty, the instruction files (`EDIT_GUIDE` and the others), `REPORT_MODEL`, `THINK_LEVEL` and the `*_THINK_LEVEL` settings: for these it means the empty string. Choice values ignore case and surrounding spaces. A value the setting cannot take stops the command at start-up with one line naming the setting, and exit status 1. This covers an unknown choice, a non-number, a number out of range (above `2000000` for `*_SEC` settings, `2000000000` for other numbers), an unclosed quote in a shell-quoted setting, and a malformed `LLM_BROKER`, `SANDBOX_MEMORY`, `SOURCE_ENCODING`, `SMOKE_FILE` or `SMOKE_ENV`. A setting given without the setting it needs (`WEB_FETCH_PROVIDER` without `WEB_SEARCH_PROVIDER`, `SMOKE_FILE` or `SMOKE_ENV` without `SMOKE_ARG`) only prints a warning. Used as a library, the module raises `codetwine_loom.SettingError`, a subclass of `ValueError`.

Some names are common words, such as `TEST_COMMAND` and `RUN_COMMAND`. Check that your CI or containers do not use them for something else.

## The minimal settings and `PRESET`

Normally these 3 are all you need to set.

| Variable | Description | Default |
|------|------|------------|
| `LLM_API_KEY` | The API key. It is sent with every model call, so it serves one provider ([several providers](#checking-the-keys-and-the-image-at-start-up)). In container runs, only the broker on the host reads it. codetwine reads the same variable | `""` |
| `PROJECT_DIR` | The target repository. `--project` gives it too (`--project .` for the current directory) | None. Every subcommand except `broker` and `setup` stops without it |
| `PRESET` | `low` / `medium` / `high`: picks one column of the table below. The `--preset` option of `run`, `answer`, `edit`, `sandbox`, `integrate`, `test-create`, `refactor` and `test-refactor` gives it too | `medium` |

`PRESET` sets only the settings that raise accuracy, cost and time together. It is not the model's effort (`THINK_LEVEL`), but a bundle that includes it. The models (`MEDIUM_MODEL`, `HIGH_MODEL`) and the turn limits (`*_MAX_TURN`) are not in it. A setting of the table that you set yourself wins for that setting only.

| Variable | `low` | `medium` | `high` | What changes |
|------|-------|----------|--------|--------------|
| `THINK_LEVEL` | `off` | `low` | `medium` | How hard one call thinks. The other stages' `*_THINK_LEVEL` follow it |
| `AGENT_MAX_TOKEN` | `16384` | `32768` | `32768` | The output limit of one call (thinking and reply together). The other stages' `*_MAX_TOKEN` follow it |
| `EXPLORE_MAX_CALL` | `3` | `4` | `5` | `explore` calls per change run |
| `SEARCH_AGAIN_MAX_CALL` | `3` | `5` | `10` | `search_again` calls |
| `WEB_SEARCH_MAX_CALL` | `10` | `15` | `20` | `web_search` calls |
| `WEB_FETCH_MAX_CALL` | `5` | `8` | `10` | `web_fetch` calls |
| `RUN_PROJECT_MAX_CALL` | `10` | `15` | `20` | `run_project` calls per stage |
| `MAX_PATCH_ROUND` | `3` | `5` | `8` | Rounds of "rewrite and test" |
| `TWO_STAGE` | `false` | `true` | `true` | Whether the planning stage runs before the change |
| `DISPATCH_MAX_TASK` | `5` | `8` | `10` | Pieces `run` splits one request into |
| `DISPATCH_MAX_RETRY` | `2` | `3` | `5` | Times a task that did not finish is handed out again |
| `DISPATCH_MAX_RERUN` | `1` | `2` | `3` | Times a request that did not succeed is run again |
| `INTEGRATE_MAX_REFUSAL` | `1` | `2` | `3` | Times the merge's `COMPLETE()` is sent back while machine checks remain |
| `INTEGRATE_MAX_RERUN` | `0` | `1` | `2` | Times an unsettled merge in `run` is redone |
| `TEST_CREATE_MAX_RERUN` | `1` | `2` | `3` | Times an unsettled test-writing stage is redone |
| `REFACTOR_MAX_RERUN` | `1` | `2` | `3` | Times an unsettled refactoring stage is redone |

All the other settings on this page normally need no change.

## Checking the keys and the image at start-up

Every subcommand that calls a model first checks that each configured model's provider has its key. These are `search`, `answer`, `edit`, `run`, `sandbox`, `integrate`, `test-create`, `refactor`, `test-refactor` and `broker`, plus `build` with `--doc-create` and `overview` when it makes the overall overview. When a key is missing, the command prints which model setting lacks which variable and stops before calling any model. A key counts as present when `LLM_API_KEY` is set or the provider's own variable (`ANTHROPIC_API_KEY` and so on) is in the shell or in `.env`. A Bedrock model (`bedrock/…`) also accepts `AWS_BEARER_TOKEN_BEDROCK` or `AWS_PROFILE`. A model name whose provider litellm cannot tell stops the command. Providers that sign in through litellm (`github_copilot/…` and so on) skip the check and sign in once. Inside a container (`LLM_BROKER` set), nothing is checked; the broker on the host holds the keys.

These need no key and call no model: `history`, `warm`, `run-tests`, `format`, `build` without `--doc-create`, `search` with `QUERY_EXPAND=false`, `overview` with `OVERVIEW=false` or `QUERY_EXPAND=false`, and `test-create --target-only`. When no `.env` was read and the command runs in a terminal, the [first-run guide](usage.md#the-first-run-guide-and-setup) asks for the missing keys instead of stopping.

**`LLM_API_KEY` does not work when the models use 2 or more providers.** Leave it empty and put each key in the provider's own variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and so on). These variables are not passed to the programs the tool starts ([Limitations](limitations.md#programs-that-are-started-get-no-keys)).

The subcommands that start containers (`sandbox`, `warm`, and `run` and `integrate` without `--here`) also check that the container engine is on `PATH`, and stop if it is not. Then they check the image; see [Building the image](container.md#building-the-image) for what happens when it is missing or stale. When the engine does not answer within 30 seconds, a warning is printed and the command goes on.

## Proxies and certificates

When the internet (the model's API, the web APIs, the package registries) is reached through a proxy, set the standard variables in the shell or in `.env`. The first-run guide asks for them too.

| Variable | Description | Default |
|------|------|------------|
| `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` | The proxy URL (`http://proxy.corp.example:3128`). The lower-case names are read too and win. Only `http://` and `https://` proxies work; `web_search` and `web_fetch` take only `http://` | Empty |
| `NO_PROXY` | Hosts reached without the proxy, separated by commas (`localhost,llm.corp.example`). Write each as a host alone, with no scheme or port; `*` only on its own. List the host of `LLM_API_BASE` when it is inside your network. The lower-case name is read too | Empty |
| `SSL_CERT_FILE` | A PEM file of certificate authorities to trust in place of the default ones, for a proxy that signs TLS with its own authority. `REQUESTS_CA_BUNDLE` is set to the same file when empty | Empty |

The values are checked before anything starts. A subcommand that calls a model stops, naming the variable, on a SOCKS proxy, a URL with a path, a non-numeric port, a non-ASCII user or password, or a `NO_PROXY` item with a port, a scheme, or `*` mixed with other items. Write the proxy's user and password in ASCII and percent-encode `#`, `/`, `?`, `[`, `]` and `\` (`%23`, `%2F`, `%3F`, `%5B`, `%5D`, `%5C`). Every subcommand stops when `SSL_CERT_FILE` is not a file or holds no PEM certificate; `~` is not expanded.

These settings apply to the model calls, `web_search` and `web_fetch`, and to the containers that fetch dependencies (`warm` and refills). Those containers get the proxy variables, including any user and password (with a warning), and the `SSL_CERT_FILE` file. Maven and Gradle get the proxy and the certificate authority through `JAVA_TOOL_OPTIONS`, without the user and password. The containers that make changes get none of these. A proxy listening on `127.0.0.1` cannot be reached from containers. For image builds behind a proxy, see [Building the image](container.md#building-the-image).

## Paths and analysis

| Variable | Description | Default |
|------|------|------------|
| `OUTPUT_DIR` | Where the analysis, the index and the run records go. `--output-dir` gives it too. Every subcommand except `format`, `setup` and `broker` without `--project` stops when it is inside `PROJECT_DIR` or contains it, cannot be written to, or holds the analysis of another repository with the same directory name | `./out` |
| `CODETWINE_MAX_WORKER` | How many files codetwine analyses in parallel | `4` |
| `BUILD_EXCLUDE` | Directory and file names left out of the analysis, on top of codetwine's own exclusions. Shell-quoted (`BUILD_EXCLUDE="vendor migrations"`); a name containing `/` is a path from the repository root. `--exclude` adds to it. Takes effect the next time codetwine runs | Empty |
| `SOURCE_ENCODING` | Encodings to try, in order, for a file with no BOM that is not UTF-8. Python codec names, separated by commas (`cp1252,iso8859_15`). The analysis and the reading and rewriting tools use them. A file none of them can read is read in the encoding codetwine guesses, with a warning | Empty |

## Index building and search

These settings tune the search that picks the code relevant to a request ([Search](processing-flow.md#3-search)).

| Variable | Description | Default |
|------|------|------------|
| `DEF_K` | Definitions kept in search step 2 | `12` |
| `INCLUDE_CONSTANT` | Whether constants, variables and similar declarations are search targets in step 2 ([which kinds](#defaults-by-language)) | `False` |
| `EXPAND_DECAY` | The factor a score is multiplied by per hop along call relations. A neighbour's share is also divided by its number of callers | `0.5` |
| `SCOPE_FILE_K` | The most files `search_scope()` returns | `10` |
| `SUMMARY_MAX_CHAR` | The most characters of one file summary in `scope["file_summaries"]`; a longer one is cut and ends with `... (N chars omitted)`. `0` does not cut | `4000` |
| `RRF_K` | The rank-fusion constant for ranks from definitions. Smaller makes rank 1 weigh more | `60` |
| `RRF_K_FILE` | The rank-fusion constant for ranks from the file index. Smaller than `RRF_K` favours the file index's top hits | `60` |
| `PATH_WEIGHT` | The weight of words from file paths. `0` leaves paths out, so files named in the request are no longer picked up. A change rebuilds the index on the next command | `3` |
| `CONFIDENCE_MIN_SCORE` | When rank 1's `raw` score (`top_raw` in the search trace) is below this, the search confidence is `low` | `0.3` |
| `QUERY_EXPAND` | Whether search step 1 turns the request into English search terms. An ASCII-only request skips it unless the overall overview is passed | `True` |
| `QUERY_EXPAND_MODEL` | The model of step 1 (litellm form) | The value of `MEDIUM_MODEL` |
| `QUERY_EXPAND_MAX_TOKEN` | The output limit of step 1 | `200` |
| `QUERY_EXPAND_WORD_RANGE` | How many words step 1 asks for. The words returned are cut at the largest number in it | `20-40` |
| `DOC_CREATE_MODEL` | The model codetwine writes design documents with (litellm form), used only with `--doc-create`. codetwine's `LLM_MODEL` is not read | The value of `MEDIUM_MODEL` |
| `DOC_CREATE_EFFORT` | How hard codetwine thinks while writing design documents: `off`, `low`, `medium` or `high`. codetwine's `DOC_EFFORT` is not read | `low` |
| `DOC_TEMPLATE_PATH` | The template for design documents, read only with `--doc-create` ([details](usage.md#doc_template_path)) | `doc_template.json` at the top of the repository this tool was installed from; codetwine's own default when that file is missing |

## The overall overview

The overall overview is a short description of the repository's requirements: its purpose and users, the inputs and outputs of each feature, the processing stages, and how it is used from outside. It names no files, functions, settings or numbers. The answering worker (`ANSWER_MODEL`) writes it once into `overview.json` in the output directory, and search step 1 reads it for every search of a run; ASCII-only requests are then rewritten too.

| Variable | Description | Default |
|------|------|------------|
| `OVERVIEW` | Whether to make the overall overview and pass it to search step 1. It is not passed when `QUERY_EXPAND` is `false`. `--no-overview` sets `false` | `True` |
| `OVERVIEW_TIMEOUT_SEC` | The time limit (seconds) for `run` and `sandbox` to make the overview, analysis included. When it runs out, the run goes on without an overview and logs a warning. `0` means no limit | `0` |

`search`, `run`, `answer`, `edit` and `sandbox` make the overview; `build` does not, and the `overview` subcommand makes only the overview. An overview is made again when files have been added or removed since it was made, or its question has changed; edits inside files do not count. `run` and `sandbox` make it in a container; when that fails, they warn and go on without it. Within one run, the overview of the starting state stays in use.

## Models and model calls

| Variable | Description | Default |
|------|------|------------|
| `MEDIUM_MODEL` | The model of the stages that carry out the work (litellm form). `EDIT_MODEL`, `ANSWER_MODEL`, `EXPLORE_MODEL`, `WEB_SEARCH_MODEL`, `QUERY_EXPAND_MODEL`, `REPORT_MODEL`, `TEST_CREATE_MODEL`, `REFACTOR_MODEL` and `DOC_CREATE_MODEL` follow it | `anthropic/claude-sonnet-5-5` |
| `HIGH_MODEL` | The model of the stages that decide what to do (litellm form). `PLAN_MODEL`, `DISPATCH_MODEL` and `INTEGRATE_MODEL` follow it | `anthropic/claude-opus-5-5` |
| `AGENT_MAX_TOKEN` | The output limit of one call of the change stage, **thinking and reply together**. The other stages' `*_MAX_TOKEN` follow it | The `PRESET` column (`32768` for medium) |
| `THINK_LEVEL` | How much one call thinks: `off`, `low`, `medium` or `high`. Empty sends no thinking request (the provider's default). The other stages' `*_THINK_LEVEL` follow it | The `PRESET` column (`low` for medium) |
| `CUT_REPLY_MAX_RETRY` | How many times a turn whose reply was cut at the output limit before any code block is called again, without using up a turn. `0` counts it as a wasted turn | `2` |
| `LLM_API_BASE` | A non-standard endpoint (Ollama and the like). It must not answer with redirects ([details](limitations.md#write-an-endpoint-that-does-not-return-redirects-in-llm_api_base)) | `""` |
| `LLM_MAX_RETRY` | How many times one call is retried. Only overloads and server errors (429, 5xx), connection failures and timeouts are retried | `3` |
| `LLM_RETRY_WAIT_SEC` | The wait (seconds) before the first retry. It doubles each time, up to `LLM_CALL_TIMEOUT_SEC`. A longer `retry-after` from the provider wins | `10` |
| `LLM_CALL_TIMEOUT_SEC` | The time limit (seconds) for one attempt of one call. Set it above the time a reply of `AGENT_MAX_TOKEN` tokens takes | `600` |
| `LLM_IDLE_TIMEOUT_SEC` | For a streamed reply, the longest wait (seconds) for the next part. `0` waits up to `LLM_CALL_TIMEOUT_SEC` | `120` |
| `PROMPT_CACHE` | Whether to mark the prompt for the provider's prompt cache. The `usage` field of the turn log shows what was cached | `true` |

`THINK_LEVEL` asks for adaptive thinking. A model that refuses it gets a thinking budget instead (1/8, 1/4 or 1/2 of the output limit for `low`, `medium` and `high`), or no thinking request when that is refused too. For models whose thinking cannot be turned off, `off` is sent as `low`. For some providers (Gemini and others) litellm drops the request, so every level behaves the same. Search step 1 always uses `off`.

When litellm knows a lower output limit for the model, that limit is used. For a model litellm does not know, a call refused for its context length is sent again with half the output limit, down to `4096` ([local model servers](limitations.md#a-local-model-server-needs-a-context-that-holds-the-prompt-and-the-reply)).

## Agents

The change stage (`edit`, `sandbox`, `run`'s change workers) and the answering worker (`answer`, `run`'s question workers) work in turns: each turn the model writes one Python block, which runs in a REPL.

| Variable | Description | Default |
|------|------|------------|
| `EDIT_MODEL` | The model of the change stage (litellm form) | The value of `MEDIUM_MODEL` |
| `RLM_MAX_TURN` | The turn limit of the change stage. The answering, planning and dispatching stages follow it. In every stage, one extra turn is given (once per run) when the last turn's `COMPLETE()` is sent back | `40` |
| `EDIT_DEADLINE_TURN` | With this many turns left, the change stage is told to finish with `COMPLETE()`. `0` does not tell it. The other stages' `*_DEADLINE_TURN` follow it | `3` |
| `EDIT_LATE_ANSWER` | When the turns run out without `COMPLETE()`, whether to ask once more what was changed. The reply is recorded; the run still counts as failed | `True` |
| `ANSWER_MODEL` | The model of the answering worker (litellm form) | The value of `MEDIUM_MODEL` |
| `ANSWER_MAX_TOKEN` / `ANSWER_THINK_LEVEL` | The output limit and thinking of the answering worker | The values of `AGENT_MAX_TOKEN` / `THINK_LEVEL` |
| `ANSWER_MAX_TURN` | The turn limit of the answering worker | The value of `RLM_MAX_TURN` |
| `ANSWER_DEADLINE_TURN` | With this many turns left, the answering worker is told to hand over its answer | The value of `EDIT_DEADLINE_TURN` |
| `ANSWER_LATE_ANSWER` | Whether an answering worker that ran out of turns is asked once more for the answer | The value of `EDIT_LATE_ANSWER` |
| `RLM_MAX_OUTPUT_CHAR` | The most characters of one turn's output kept in the history. The model is told this number. `0` does not cut | `40000` |
| `HISTORY_MAX_CHAR` | The most characters of history put into the prompt. Above it, the output of the oldest turns is folded first, until the history fits in half the limit. `0` does not fold. Lower it for models with a small context | `600000` |
| `HISTORY_KEEP_TURN` | The number of latest turns never folded | `3` |
| `HISTORY_FOLD_HEAD_CHAR` | The leading characters kept of a folded turn | `2000` |

When the provider refuses a prompt as too long for its context, the history limit is halved for the rest of that stage and the prompt is sent once more.

## Dispatching and the report (`run`)

`run` first splits the request into change tasks or sub-questions, or returns questions for you to answer. Each piece runs as its own agent ([How `run` handles a request](usage.md#how-run-handles-a-request)).

| Variable | Description | Default |
|------|------|------------|
| `DISPATCH_MODEL` | The model of the dispatching stage (litellm form) | The value of `HIGH_MODEL` |
| `DISPATCH_MAX_TOKEN` | The output limit of one call of the dispatching stage | The value of `AGENT_MAX_TOKEN` |
| `DISPATCH_THINK_LEVEL` | How hard the dispatching stage thinks | The value of `THINK_LEVEL` |
| `DISPATCH_MAX_TURN` | The turn limit of the dispatching stage | The value of `RLM_MAX_TURN` |
| `DISPATCH_DEADLINE_TURN` | With this many turns left, the dispatching stage is told to hand over. `0` does not tell it | The value of `EDIT_DEADLINE_TURN` |
| `DISPATCH_MAX_TASK` | The most pieces one request is split into. Each piece is one agent and one container | The `PRESET` column (`8` for medium) |
| `DISPATCH_TASK_MAX_CHAR` | The most characters of one task's text. A longer one is sent back once, then cut. `0` means no limit | `4000` |
| `DISPATCH_MAX_UNCLEAR` | The most questions the dispatching stage can return to you instead of splitting. `run` then starts no agent, prints them and exits with code 1. `0` does not allow returning questions | `3` |
| `DISPATCH_ANSWER_MAX_CHAR` | The most characters of an answer: one worker's, the combined one, and the report's prose. A longer one is sent back once to be shortened, then cut. `0` means no limit | `6000` |
| `DISPATCH_PARALLEL` | How many agents run at the same time. **Budgets and memory limits apply per agent**, so N agents use N times as much | `3` |
| `DISPATCH_MAX_RETRY` | How many times a task that did not finish (or a sub-question with no answer) is handed out again, each time on a new branch. `0` does not retry | The `PRESET` column (`3` for medium) |
| `DISPATCH_MAX_RERUN` | How many times **a request that did not succeed** (non-zero exit code) is run again from the branches it left. `0` runs it once | The `PRESET` column (`2` for medium) |
| `DISPATCH_KEEP_RUN` | How many resumable runs are kept per command, newest first. At the end of `run`, `sandbox` and `integrate`, the branches and output directories of older ones are removed. `0` removes nothing | `3` |
| `MAX_TOKEN_PER_RUN` | The most tokens (input and output) one run of `run`, `sandbox` or `integrate` may use across all its agents, the overview and the report. `0` means no limit | `0` |
| `REPORT_MODEL` | The model that writes the closing report on what was changed where. **Empty writes no prose**, only the list of files | The value of `MEDIUM_MODEL` |
| `REPORT_MAX_TOKEN` | The output limit of the report's call | The value of `AGENT_MAX_TOKEN` |
| `REPORT_DIFF_MAX_CHAR` | The most characters of the diff given to the report's model. `0` means no limit | `60000` |

Tasks that name the same definition run one after another, each starting from the commit the previous one left. A task that hit `SANDBOX_MEMORY` (exit status 137) is not handed out again. Each rerun of `DISPATCH_MAX_RERUN` splits, works and merges again, and reruns stop at the first success; `runs: N` appears at the top of the output. A request also counts as not succeeded when the written tests found cases the change does not do. There is no rerun when the union was not applied because of the paths it touches ([Before a change goes into the working tree](container.md#before-a-change-goes-into-the-working-tree)), when the merge cannot settle because of the tests on `HEAD`, or after exit status 137.

A resumable run is one that left something to carry on from: an unfinished request, a failed `sandbox` branch, an uncommitted patch (such as one from `sandbox --dry-run`), or an unsettled union. Runs that went all the way through, runs in progress, runs named by `--resume` or `--branch`, and other repositories' runs are not counted or removed.

When `MAX_TOKEN_PER_RUN` is reached, no new container starts, further model calls are refused, and the run exits with code 1, keeping the task branches. When it is reached after the merge, changes already applied stay and a `token limit: …` line appears. Tokens spent by `--doc-create` are not counted ([Cost limits](limitations.md#cost-limits)).

## Merging (`integrate`)

`integrate` combines several branches into one ([Combining several branches into one](container.md#combining-several-branches-into-one-integrate)). The same settings apply to the merge stage of `run` and to the worker that combines the answers to a question.

| Variable | Description | Default |
|------|------|------------|
| `INTEGRATE_MODEL` | The model of the merge stage and of the worker that combines answers (litellm form) | The value of `HIGH_MODEL` |
| `INTEGRATE_MAX_TOKEN` | The output limit of one call | The value of `AGENT_MAX_TOKEN` |
| `INTEGRATE_THINK_LEVEL` | How hard it thinks | The value of `THINK_LEVEL` |
| `INTEGRATE_MAX_TURN` | The turn limit. It does not follow `RLM_MAX_TURN` | `50` |
| `INTEGRATE_DEADLINE_TURN` | With this many turns left, it is told to finish. `0` does not tell it | The value of `EDIT_DEADLINE_TURN` |
| `INTEGRATE_WORK_MAX_ITEM` | The most items in each list of the merge's work list (conflicted files, broken references, failing tests and so on). A cut list sets `complete` to `false`. The same number caps the lists in every stage's state and the items `record_decision` and `record_gap` can record | `40` |
| `INTEGRATE_MAX_REFUSAL` | How many times `COMPLETE()` is sent back while machine checks remain ([the checks](processing-flow.md#7-merging-branches-integrate)). `0` accepts the first `COMPLETE()` and records what remains | The `PRESET` column (`2` for medium) |
| `INTEGRATE_MAX_RERUN` | In `run`, when every worker finished but the merge ended unsettled, how many times only the merge is redone. `0` does not redo it | The `PRESET` column (`1` for medium) |

A redo of `INTEGRATE_MAX_RERUN` reuses the workers' branches and stops when the merge settles (there is no redo after exit status 137); `merges: N` appears at the top of the output. When it is still unsettled at the limit, `run` goes on to the reruns of `DISPATCH_MAX_RERUN`. Some red test runs on `HEAD` keep every merge from settling ([which ones](limitations.md#the-test-gate-of-integration-compares-with-the-merge-target)); the `NOT applied to the working tree:` line says what to fix first. For a question, the combining worker is redone with the same answers (`combines: N`); when it still returns nothing, the workers' answers are printed as they are and the exit code is 1.

## Edit history

| Variable | Description | Default |
|------|------|------------|
| `EDIT_HISTORY` | Whether each run records what was asked, what was done and which files were touched in `<output-dir>/<project-name>/edit_history.json`, and puts an unfinished previous run at the top of the next request. With `false`, `edit` records nothing and does not offer to carry on | `True` |
| `EDIT_HISTORY_KEEP` | The number of entries kept; the oldest go first | `5` |
| `EDIT_HISTORY_REQUEST_CHAR` | The characters of the request kept per entry. `0` does not cut | `400` |
| `EDIT_HISTORY_DID_CHAR` | The characters of "what was done" kept per entry. Each `record_decision` and `record_gap` item and each test case is cut at the same length. `0` does not cut | `800` |
| `EDIT_HISTORY_MAX_FILE` | The number of files kept per entry, for each of added, changed and deleted | `30` |

Changes inside a container (`sandbox`, `run`'s workers) always write this record, starting fresh each time. The format is in [output.md](output.md#edit_historyjson).

## Writing tests after the merge (`TEST_CREATE`)

After the merge of `run` or `integrate` settles, a test-writing stage writes tests for the changed definitions before anything goes into the working tree. It adds a separate commit (`test-create: ...`) to the same branch. How it works is in [Writing tests after the merge](processing-flow.md#8-writing-tests-after-the-merge-test-create), and what it checks in [Limitations](limitations.md#what-the-stages-after-the-merge-check).

| Variable | Description | Default |
|------|------|------------|
| `TEST_CREATE` | Whether to run the test-writing stage. `--no-test-create` of `run` and `integrate` sets `false` | `True` |
| `TEST_CREATE_MODEL` | The model of its two parts (litellm form) | The value of `MEDIUM_MODEL` |
| `TEST_CREATE_MAX_TURN` | The turn limit of each part | `50` |
| `TEST_CREATE_MAX_TOKEN` | The output limit of one call | The value of `AGENT_MAX_TOKEN` |
| `TEST_CREATE_THINK_LEVEL` | How hard this stage thinks | The value of `THINK_LEVEL` |
| `TEST_CREATE_MAX_DEFINITION` | About how many changed definitions one test task (one container) covers. Above it, the work is split into several test tasks. `0` does not split | `10` |
| `TEST_CREATE_MAX_REFUSAL` | How many times `COMPLETE()` is sent back while checks remain. `0` accepts the first one | The value of `INTEGRATE_MAX_REFUSAL` |
| `TEST_CREATE_MAX_RERUN` | How many times an unsettled test-writing stage is redone in the same run, keeping what the settled test tasks wrote. `0` does not redo it | The `PRESET` column (`2` for medium) |

The stage also runs in a repository with no tests, and puts the first tests under `TEST_PATH`. Test cases are numbered `case_<number>`. A test task may write only the implementation files it covers, the tests that refer to them, new test files, and the primary language's dependency files (`pyproject.toml`, `pom.xml`, `package.json` and so on). Implementation and dependency files it changed are listed under `also changed (not tests):`. The written tests are also run on the code before the change, and the result appears as `on the base:` (`red`, `green` or `unknown`).

The stage ends without calling the model, with a `tests: none written (...)` line, when there is no test command, when the repository has no tests and its primary language has no dependency file (SQL, COBOL, BMS, JCL, CSD), when the change touches only tests, or when the starting test result cannot be read (the tests do not run, do not name the failed tests, or stop early, as with pytest's `-x`).

**Cases the change does not do.** When the code does not behave as a test case says and the stage cannot fix it within its files, it records the case. The cases appear under `NOT done by the change, as the tests found (<N>):`, and `run` and `integrate` exit with code 1 and keep the union branch. With `DISPATCH_MAX_RERUN` at 1 or more, `run` runs the request again with those cases.

**When the stage ends unsettled**, it is redone up to `TEST_CREATE_MAX_RERUN` times for the unsettled files only. There is no redo after exit status 137, or when no unsettled test task has its test command installed (`sh: 1: bun: not found` and the like). When it is still unsettled, only the merged change is applied to the working tree and the exit code does not change. Passing the name on the last line to `run --resume` or `integrate --resume` writes only the tests again.

## Refactoring after the merge (`REFACTOR` / `TEST_REFACTOR`)

After the test-writing stage, two stages refactor only what the request and the test-writing stage changed: **implementation refactoring** (`refactor`), then **test refactoring** (`test-refactor`). Each adds a separate commit (`refactor: ...` / `test-refactor: ...`) to the same branch. How they work is in [Refactoring after the merge](processing-flow.md#9-refactoring-after-the-merge-refactor-test-refactor).

| Variable | Description | Default |
|------|------|------------|
| `REFACTOR` | Whether to run the implementation refactoring stage. `--no-refactor` sets `false` | `True` |
| `TEST_REFACTOR` | Whether to run the test refactoring stage. `--no-test-refactor` sets `false` | `True` |
| `REFACTOR_MODEL` | The model of both stages (litellm form) | The value of `MEDIUM_MODEL` |
| `REFACTOR_MAX_TURN` | The turn limit of each stage | `50` |
| `REFACTOR_MAX_TOKEN` | The output limit of one call | The value of `AGENT_MAX_TOKEN` |
| `REFACTOR_THINK_LEVEL` | How hard the stages think | The value of `THINK_LEVEL` |
| `REFACTOR_MAX_REFUSAL` | How many times `COMPLETE()` is sent back while checks remain. `0` accepts the first one | The value of `INTEGRATE_MAX_REFUSAL` |
| `REFACTOR_MAX_RERUN` | How many times an unsettled refactoring stage is redone in the same run, from the commit it started at (counted per stage). `0` does not redo it | The `PRESET` column (`2` for medium) |

Implementation refactoring may rewrite only the non-test files whose definitions the request or the test-writing stage changed; it adds no files. Test refactoring may rewrite the test files the request and the test-writing stage added or changed, and add test files. Without an instruction file (`REFACTOR_GUIDE`, `TEST_REFACTOR_GUIDE`; see [below](#the-instruction-files-for-changes-codetwine_loommd)), a stage looks for code that does not match the surrounding style, duplication, and things left unused by the request. It does not change behaviour, names and signatures used elsewhere, or lines the request did not touch; test refactoring keeps what each test checks, the test names and the case numbers. A stage with no target, no test command or no readable starting test result ends without calling the model and prints `refactoring: nothing done (...)` or `test refactoring: nothing done (...)`.

When a stage ends unsettled, its commit is dropped and it is redone up to `REFACTOR_MAX_RERUN` times (not after exit status 137, or when its test command is not installed). When it is still unsettled, the later stages do not run, and the working tree gets the result of the last settled stage. The exit code does not change. Passing the name on the last line to `run --resume` or `integrate --resume` runs again from that stage.

## The instruction files for changes (`CODETWINE_LOOM.md`)

| Variable | Description | Default |
|------|------|------------|
| `EDIT_GUIDE` | The instruction file for the planning and change stages. A relative path is from the target repository root; an absolute path is used as is. `""` reads no file. `edit --guide` gives it too | `CODETWINE_LOOM.md` |
| `TEST_GUIDE` | The instruction file for writing tests (the framework, where fixtures and helpers go, naming, what tests must not do). Read like `EDIT_GUIDE` | `CODETWINE_LOOM_TEST.md` |
| `REFACTOR_GUIDE` | The instruction file for implementation refactoring. Read like `EDIT_GUIDE` | `CODETWINE_LOOM_REFACTOR.md` |
| `TEST_REFACTOR_GUIDE` | The instruction file for test refactoring. Read like `EDIT_GUIDE` | `CODETWINE_LOOM_TEST_REFACTOR.md` |
| `EDIT_GUIDE_MAX_CHAR` | The most characters of an instruction file put into the instructions; the rest is cut and marked as cut. `0` does not cut | `40000` |

**By default, how to make a change is left to the model.** Your conventions, what to run before finishing, and what must not be touched reach the model only through the request or an instruction file. A `CODETWINE_LOOM.md` at the top of the target repository is added to the instructions of the planning and change stages, marked as the repository's own instructions. `run`'s workers get it; the dispatching stage does not.

Each stage reads only its own file. All files are optional; a missing file adds nothing, and no other file stands in for it. To have several stages follow the same conventions, write them in each file.

| Stage | File read | Setting |
|---|---|---|
| The planning and change stages (`edit`, `sandbox`, `run`'s workers) | `CODETWINE_LOOM.md` (how to make changes, including test conventions the workers should keep) | `EDIT_GUIDE` |
| The test-writing stage's second part, which writes the code | `CODETWINE_LOOM_TEST.md` (how to write tests) | `TEST_GUIDE` |
| Implementation refactoring | `CODETWINE_LOOM_REFACTOR.md` | `REFACTOR_GUIDE` |
| Test refactoring | `CODETWINE_LOOM_TEST_REFACTOR.md` | `TEST_REFACTOR_GUIDE` |
| Dispatching, answering questions, merging, and the test-writing stage's first part | None | — |

```markdown
# How this repository is changed

- Add one line at the top of `CHANGELOG.md` for changes users can see
- Do not call `print()` from library code. Log with the module's `logger`
- Do not add new third-party libraries
```

When a file disagrees with this tool's own instructions, **the tool's instructions win**. `edit` on the host can read a file outside the repository (an absolute path, or `../shared/AGENT.md`). Containers (`sandbox`, `run`, `integrate`) see only their clone, so keep the file inside the repository for them ([details](limitations.md#the-base-is-the-working-tree-itself)). Only regular files are read. A non-default file that is missing prints a warning. **`AGENTS.md` is not read by default**; set `EDIT_GUIDE=AGENTS.md` to use it.

## The planning stage (two stages)

With `TWO_STAGE=true`, a planning stage runs before the change. It cannot rewrite the repository, but it can run the program with `run_project` when there is a `RUN_COMMAND`. Only its plan reaches the change stage: the files to touch, what to do in each and in what order, what passes between files, the facts it confirmed, and how to check. The change stage sees each turn which planned files it has touched. When no readable plan comes out, the change stage runs without one.

| Variable | Description | Default |
|------|------|------------|
| `TWO_STAGE` | Whether the planning stage runs before the change | The `PRESET` column (`true` for medium) |
| `PLAN_MODEL` | The model of the planning stage (litellm form) | The value of `HIGH_MODEL` |
| `PLAN_MAX_TURN` | The turn limit of the planning stage | The value of `RLM_MAX_TURN` |
| `PLAN_MAX_TOKEN` | The output limit of one call | The value of `AGENT_MAX_TOKEN` |
| `PLAN_THINK_LEVEL` | How hard the planning stage thinks | The value of `THINK_LEVEL` |
| `PLAN_DEADLINE_TURN` | With this many turns left, it is told to finish with `COMPLETE()`. `0` does not tell it | The value of `EDIT_DEADLINE_TURN` |
| `PLAN_MAX_FILE` | The most files one plan can list | `8` |
| `PLAN_MAX_FACT` | The most confirmed facts one plan can carry | `12` |
| `PLAN_ANSWER_MAX_CHAR` | The most characters of the plan (JSON); the rest is cut. `0` means no limit | `40000` |
| `PLAN_COMPLETE_GUARD` | Whether the change stage's `COMPLETE()` is sent back once while planned files are untouched | `true` |
| `PLAN_LATE_ANSWER` | When the planning stage runs out of turns without `COMPLETE()`, whether to ask once more for the plan | `true` |

## The explorer

The change stage can call `explore("<aspect>")`. This starts a separate agent with read-only tools that returns only candidate places to change (`file`, `lines`, `signature`, `why`). Its turns do not count toward `RLM_MAX_TURN`, and its history is not passed back.

| Variable | Description | Default |
|------|------|------------|
| `EXPLORE_MODEL` | The model of the explorer (litellm form) | The value of `MEDIUM_MODEL` |
| `EXPLORE_MAX_TOKEN` | The output limit of one call | The value of `AGENT_MAX_TOKEN` |
| `EXPLORE_THINK_LEVEL` | How hard the explorer thinks | The value of `THINK_LEVEL` |
| `EXPLORE_MAX_TURN` | The turn limit of one explorer. When it runs out, the explorer is asked once more for what it found | `20` |
| `EXPLORE_MAX_CALL` | `explore` calls per change run (per task in `run`). `0` does not give the tool | The `PRESET` column (`4` for medium) |
| `EXPLORE_ANSWER_MAX_CHAR` | The most characters of the explorer's result (JSON); the rest is cut. `0` means no limit | `40000` |
| `EXPLORE_MAX_RESULT` | The number of candidates one explorer returns | `6` |
| `SEARCH_AGAIN_MAX_CALL` | `search_again` calls. In a change run, the planning stage and the explorers share one budget; the answering and dispatching stages get their own each time. `0` does not give the tool | The `PRESET` column (`5` for medium) |

## Reading tools (`read_source_file`, `grep_source`)

| Variable | Description | Default |
|------|------|------------|
| `READ_MAX_CHAR` | The most characters `read_source_file` returns at once, cut at a line end with a note on how to read on. Never more than `RLM_MAX_OUTPUT_CHAR` | `40000` |
| `GREP_MAX_HIT` | The most lines `grep_source` returns, further limited by what fits in `RLM_MAX_OUTPUT_CHAR` | `200` |
| `GREP_MAX_HIT_COUNT` | The most lines `grep_source` counts in `counts` | `5000` |
| `GREP_CONTEXT_LINE` | Lines of context before and after each of the first hits. `0` adds none | `3` |
| `GREP_CONTEXT_HIT` | How many hits get context lines; later hits show only `path`, `line` and `text` | `5` |
| `GREP_HIT_OVERHEAD_CHAR` | The estimated characters per hit besides its text, used to work out how many hits fit | `60` |
| `GREP_LINE_MAX_CHAR` | The characters kept per line; a cut line ends with `... (N chars omitted)`. `0` does not cut | `200` |
| `GREP_SKIP_DIR` | Directory names `grep_source` skips, separated by commas. The primary language's build output directories ([Defaults by language](#defaults-by-language)) are skipped too | `__pycache__,.git,.github,.venv,venv,.tox,.mypy_cache,.pytest_cache,node_modules,dist,build,out,target,.gradle,.mvn,vendor,.idea,.vscode` |

## Asking about facts outside the repository (`web_search`)

With a search provider set, the change, planning, answering and dispatching stages can call `web_search("<one question in English>")`. It runs one search and returns the top results (title, URL, excerpt) as they are. **Without a search provider, the tool is not given.** There are two kinds of provider: **the model provider's own search** (`WEB_SEARCH_PROVIDER=model`, using the LLM key) or **a search API** (a preset name in `WEB_SEARCH_PROVIDER` and its key in `WEB_SEARCH_API_KEY`). In container runs, the broker on the host sends the searches; the provider settings and the key never enter a container ([limitations](limitations.md#web_search-needs-a-search-provider-and-returns-snippets-only)).

| Variable | Description | Default |
|------|------|------------|
| `WEB_SEARCH_PROVIDER` | `model`, or a preset name from the table below. An unknown name stops the command at start-up | Empty (no `web_search`) |
| `WEB_SEARCH_API_KEY` | The key of the search API. A preset without a key stops the command at start-up. Not needed for `model` | Empty |
| `WEB_SEARCH_MAX_CALL` | `web_search` calls. In a change run, the planning and change stages share one budget; the answering and dispatching stages get their own each time. `0` does not give the tool | The `PRESET` column (`15` for medium) |
| `WEB_SEARCH_MODEL` | The model that searches with `model` (litellm form) | The value of `MEDIUM_MODEL` |
| `WEB_SEARCH_MAX_TOKEN` | With `model`, the output limit of its answer | The value of `AGENT_MAX_TOKEN` |
| `WEB_SEARCH_CONTEXT_SIZE` | With `model`, how much of the pages the model reads before answering (`low` / `medium` / `high`, on the provider's scale) | `low` |
| `WEB_SEARCH_MAX_RESULT` | The number of results returned (with `model`, its answer counts as one) | `10` |
| `WEB_SEARCH_RESULT_MAX_CHAR` | The most characters of the whole result list. Each result is cut at this divided by `WEB_SEARCH_MAX_RESULT`. `0` means no limit | `40000` |
| `WEB_SEARCH_QUESTION_MAX_CHAR` | The most characters of the question. A longer question is refused without using up a call | `300` |
| `WEB_SEARCH_TIMEOUT_SEC` | The time limit (seconds) of one search. When the provider asks to wait longer than this, the search is refused | `30` |
| `WEB_SEARCH_MIN_INTERVAL_SEC` | The shortest interval (seconds) between searches from one process. `Retry-After` of `429` / `503` is honoured | `1.0` |
| `WEB_SEARCH_MAX_BYTE` | The most bytes read from a response; a larger one is refused | `1048576` |

| `WEB_SEARCH_PROVIDER` | Search provider | Key | Charges and terms of use |
|---|---|---|---|
| `model` | One model call with web search on, through the provider of `WEB_SEARCH_MODEL`: Anthropic, OpenAI or Gemini | `LLM_API_KEY` only | The model provider charges for the searches and the tokens |
| `tavily` | Tavily Search API (`POST https://api.tavily.com/search`) | Issued at app.tavily.com | Follow the contract of the key's issuer |
| `brave` | Brave Search API (`GET https://api.search.brave.com/res/v1/web/search`) | Issued at api-dashboard.search.brave.com | Follow the contract of the key's issuer. **Storing search results needs a plan with storage rights** (Brave's FAQ); the turn records keep the results the model was shown |
| `ollama` | Ollama Web Search (`POST https://ollama.com/api/web_search`) | Issued with an ollama.com account (the same key as the fetch provider `ollama`) | Follow the contract of the key's issuer |

```bash
# Use only the LLM's key
export WEB_SEARCH_PROVIDER=model

# Use a search API
export WEB_SEARCH_PROVIDER=tavily
export WEB_SEARCH_API_KEY="tvly-..."
```

With `model`, the model searches, reads and answers: the first result is its answer and the rest are the pages it cited. Providers without web search, and local models, return `ok=False`. With a search API, no model takes part, so the tool works the same with any model. To turn the tool off while a provider is set, use `WEB_SEARCH_MAX_CALL=0`.

## Reading a page of the search results (`web_fetch`)

With a fetch provider and a search provider set, the same stages can call `web_fetch(<url of a search result>)`. It fetches one page through a page-reading API and returns its body as text. **Only URLs that `web_search` returned in the same stage can be read**; other URLs are refused without using up a call. The tool never fetches pages directly. The model is told that the body is text a site published, not instructions.

| Variable | Description | Default |
|------|------|------------|
| `WEB_FETCH_PROVIDER` | A preset name from the table below. An unknown name stops the command at start-up | Empty (no `web_fetch`) |
| `WEB_FETCH_API_KEY` | The key of the page-reading API. A preset without a key stops the command at start-up | The value of `WEB_SEARCH_API_KEY` when `WEB_FETCH_PROVIDER` is the same as `WEB_SEARCH_PROVIDER`, otherwise empty |
| `WEB_FETCH_MAX_CALL` | `web_fetch` calls, shared like `WEB_SEARCH_MAX_CALL`. `0` does not give the tool | The `PRESET` column (`8` for medium) |
| `WEB_FETCH_RESULT_MAX_CHAR` | The most characters of the body returned; the rest is cut and marked `... (N chars omitted)`. `0` means no limit | `40000` |
| `WEB_FETCH_TIMEOUT_SEC` | The time limit (seconds) of one fetch | `30` |
| `WEB_FETCH_MIN_INTERVAL_SEC` | The shortest interval (seconds) between fetches from one process, counted apart from searches | `1.0` |
| `WEB_FETCH_MAX_BYTE` | The most bytes read from a response; a larger one is refused | `4194304` |

| `WEB_FETCH_PROVIDER` | Fetch provider | Key | Charges and terms of use |
|---|---|---|---|
| `tavily` | Tavily Extract API (`POST https://api.tavily.com/extract`) | `WEB_SEARCH_API_KEY` when `WEB_SEARCH_PROVIDER=tavily` too, otherwise `WEB_FETCH_API_KEY` | Follow the contract of the key's issuer |
| `ollama` | `web_fetch` of Ollama Web Search (`POST https://ollama.com/api/web_fetch`) | `WEB_SEARCH_API_KEY` when `WEB_SEARCH_PROVIDER=ollama` too, otherwise `WEB_FETCH_API_KEY` | Follow the contract of the key's issuer |
| `jina` | Jina Reader API (`POST https://r.jina.ai/`) | Issued at jina.ai; put it in `WEB_FETCH_API_KEY` | Follow the contract of the key's issuer |

```bash
# Tavily for both searching and fetching, with one key
export WEB_SEARCH_PROVIDER=tavily
export WEB_SEARCH_API_KEY="tvly-..."
export WEB_FETCH_PROVIDER=tavily

# Brave for searching, Jina for fetching
export WEB_SEARCH_PROVIDER=brave
export WEB_SEARCH_API_KEY="..."
export WEB_FETCH_PROVIDER=jina
export WEB_FETCH_API_KEY="jina_..."
```

## Change tools

| Variable | Description | Default |
|------|------|------------|
| `MAX_PATCH_ROUND` | The most rounds of "rewrite, then `run_test`". The rewriting tools (`replace_in_file`, `replace_line_range`, `create_file`, `delete_file`) enforce it. `0` means no limit | The `PRESET` column (`5` for medium) |
| `STAT_ECHO_LINE` | Lines before the replaced range that `replace_line_range` echoes back in its `stat` | `3` |
| `STAT_ECHO_CHAR` | The characters kept per echoed line | `60` |
| `TURN_TIMEOUT_SEC` | The time limit (seconds) for running one turn's code. A turn over it gets `TimeoutError` as its output and the run goes on. Turns inside `explore()` count toward the outer turn. `0` means no limit | `TEST_TIMEOUT_SEC + RUN_TIMEOUT_SEC` (`420` by default) |
| `FORMAT_COMMAND` | A formatter or code generator a container runs at the end of each run on the files it added or changed (`FORMAT_COMMAND="ruff format"`). It runs once from the repository root with the paths appended, and its rewrites go into the patch. The time limit is `TEST_TIMEOUT_SEC`; a non-zero exit only warns. The command must be in the image. Not run by `edit` on the host | Empty (not run) |

## Tests and the static check

| Variable | Description | Default |
|------|------|------------|
| `TEST_COMMAND` | The command `run_test` runs, split like a shell command line. When set, only this command runs | The primary language's default, plus the other languages' defaults (see below) |
| `TEST_PATH` | Where `run_test` runs when no `paths` are given, and where the test-writing stage puts the first tests. Relative to the repository root. Files under it, and files named like the primary language's test files, count as tests. While it is unset, files under the language's other test directories at the root count too. With `.`, only file names decide, so for languages without test-file names (Python, Java, Kotlin, C / C++, Rust) the test-writing stage writes nothing | Of the primary language's test directories ([Defaults by language](#defaults-by-language)), the one with the most source files; the first one on a tie or when none exists |
| `TEST_TARGET_STYLE` | How `paths` are passed to the test command: `path`, `separator` or `none` ([Defaults by language](#defaults-by-language)). Set it together with `TEST_COMMAND` when you switch to another build tool | The primary language's default |
| `RUN_TEST` | Whether `run_test()` is given to the model. With `false`, the three stages after the merge end without calling the model | `True` |
| `TEST_TIMEOUT_SEC` | The time limit (seconds) for one test run. When it runs out, every process the test command started is stopped; processes left behind after it ends are stopped too | `300` |
| `TEST_OUTPUT_TAIL` | The characters from the end of the output that `run_test` returns; a cut output starts with a `... (N chars omitted)` line. `0` does not cut | `40000` |
| `FAIL_EXCERPT_CHAR` | The characters from the end of `failed_excerpt` (the failures only). `0` does not cut. For Rust, progress lines, passing targets and warnings are removed first | `800` |
| `CHECK_COMMAND` | The static check run by `edit` and `sandbox` (unless `--no-check` is given), the merge, the test-writing stage and the refactoring stages. Findings that were not there at the start leave the stage unsettled. When set, only this command runs | The primary language's default, plus the other languages' defaults (see below) |
| `COBOL_DIALECT` | The dialect the COBOL static check uses, passed to `cobc -std=` (`ibm`, `mf` and so on). When unset in a repository with COBOL files, `run` and `integrate` ask once in a terminal and add the answer to `.env` | Empty (GnuCOBOL's default) |
| `COBOL_COPYBOOK_DIR` | Extra directories, relative to the repository and separated by commas, where the COBOL static check looks for copybooks. Copybooks that `COPY` statements lead to are found without it. Copybooks outside the repository must be copied in | Empty |

When `TEST_COMMAND` is unset, the primary language's default runs first. Then the default test command of each other language in the index runs, when the command is on `PATH` and the language's dependency file (`package.json`, `pom.xml`, `Cargo.toml` and so on) is in the repository ([details](limitations.md#tests-of-languages-other-than-the-main-one-run-only-where-that-languages-dependency-file-is)). The combined result is `failed` when any run is red, `did not run` when one could not run, and `passed` otherwise. When no command can run, `run_test` is not given to the model. How results are read is in [How test results are read](limitations.md#how-test-results-are-read).

`CHECK_COMMAND` is split like `TEST_COMMAND`. Each output line of the form `<path>:<line>[:<column>]: <text>` or `<path>(<line>,<column>): <text>` is one finding. When it is unset, the primary language's default runs, plus the defaults of the other languages in the index whose commands are on `PATH`. When the command cannot start, exits with 127, or fails without printing a finding, a warning is printed and nothing is checked. The defaults are `ruff check --select F821` for Python, and `ctloom-check-js`, `ctloom-check-ts`, `ctloom-check-c` (C / C++), `ctloom-check-rs`, `ctloom-check-cs`, `ctloom-check-cbl` and `ctloom-check-r`, which are in the container image. Java, Kotlin, SQL, BMS, JCL and CSD have none. What each finds is in [Limitations](limitations.md#the-default-static-checks-differ-in-strength-by-language).

## Running the program (`run_project`)

| Variable | Description | Default |
|------|------|------------|
| `RUN_COMMAND` | The start of the command `run_project` runs; the model passes only the arguments after it. Without a trailing `--`, leave the program out (`RUN_COMMAND=node`): the model's first argument names the program, and calls with no arguments or starting with a flag are refused. End it with `--` only when it already names the program and the launcher takes `--` itself (`cargo run --quiet --`, `npm run start --`). When it is empty and the language has no default, `run_project` is not given | The primary language's default: `uv run` for Python, `cargo run --quiet --` for Rust, `Rscript` for R, and `ctloom-run-c` for C / C++ when the repository's top has a makefile or `configure` (the arguments are make targets, run on a copy of the repository). None for other languages |
| `RUN_TIMEOUT_SEC` | The time limit (seconds) for one `run_project` run. When it runs out, every process the program started is stopped; processes left behind after it ends are stopped too | `120` |
| `RUN_PROJECT_MAX_CALL` | `run_project` calls per stage (the planning and change stages each have their own). What is left shows in `calls_left`. `0` does not give the tool | The `PRESET` column (`15` for medium) |
| `RUN_OUTPUT_TAIL` | The characters from the end of the output that `run_project` returns; a cut output starts with a `... (N chars omitted)` line. `0` does not cut | `40000` |
| `RUN_MAX_FILE` | The most files `run_project` lists in `produced`; the number left out goes into `produced_more` | `40` |

## Automatic check after each edit (`SMOKE_ARG`)

With `SMOKE_ARG` set, **after every turn that changed the request's files since the last check, the tool starts the program with `RUN_COMMAND` + `SMOKE_ARG` and adds the result to that turn's output**. The model cannot skip it. A first start is made before any change; when the program already fails there, the check is turned off. It shows only whether the program runs, not whether the change is right. It does not run with `--dry-run`.

| Variable | Description | Default |
|------|------|------------|
| `SMOKE_ARG` | The arguments after `RUN_COMMAND`, split like a shell command line. `{workspace}` is replaced with a temporary working directory. **Empty means no check.** With `RUN_COMMAND` empty, it warns and does not check | Empty |
| `SMOKE_FILE` | Sample files written into the working directory before the start, as JSON `{"path": "contents"}` | `{}` |
| `SMOKE_ENV` | Environment variables added for the start, as JSON `{"name": "value"}` | `{}` |
| `SMOKE_REPEAT` | Starts per check. From the second start on, the working directory is not cleared | `1` |
| `SMOKE_OUTPUT_TAIL` | The characters from the end of the output added to the turn on failure. `0` does not cut | `40000` |
| `SMOKE_COMPLETE_GUARD` | Whether `COMPLETE()` is sent back once while the check fails | `true` |

```bash
SMOKE_ARG='main.py --project-dir {workspace}/sample --output-dir {workspace}/out' \
SMOKE_ENV='{"ENABLE_LLM_DOC": "false"}' \
SMOKE_FILE='{"sample/a.py": "def f():\n    return 1\n"}' \
codetwine-loom edit --project ../codetwine --output-dir ./out \
  --request "Reuse the previous result for files that have not changed"
```

## Sandbox

What the container may and may not reach is in [The boundary `sandbox` sets up](container.md#the-boundary-sandbox-sets-up).

| Variable | Description | Default |
|------|------|------------|
| `CONTAINER_ENGINE` | The container engine. `--engine` gives it too. Empty looks for `podman`, then `docker`, on `PATH` | Empty |
| `CONTAINER_IMAGE` | The image the containers run. `--image` gives it too ([Building the image](container.md#building-the-image)) | `codetwine-loom`, which holds the toolchains of every language |
| `SANDBOX_MEMORY` | The memory the container's processes may use, in the engine's `--memory` form (`2g`, `512m`). The clone and temporary files are on the host's disk and do not count. Build parallelism is set to this value in whole GiB (at least 1). A container that reaches it ends with exit status 137 | `2g` |
| `SANDBOX_MAX_PROCESS` | The most processes (threads included) one container may hold at once | `4096` |
| `SANDBOX_TIMEOUT_SEC` | The time limit (seconds) for one agent's container. A container that runs out is removed and counts as not finished (`run` hands it out again under `DISPATCH_MAX_RETRY`). **`0` means no limit** | `0` |
| `SANDBOX_NETWORK` | A container network the run's containers join, for tests that need a service in another container ([how to use it](#repositories-whose-tests-need-another-service-sandbox_network)). **Only networks the engine reports as internal** are accepted; anything else stops the command before containers start | Empty (no network) |
| `SANDBOX_CACHE_DIR` | Where the package cache lives, shared read-write by every container. Set it to share one cache between runs with different output directories. It must not be in the target repository, be the output directory or above it, or be in the output directory's `<repo-name>/` or `agents/` | Empty (`<output-dir>/agents/.repo/<repo-name>/cache`) |
| `WARM_MAX_REFILL` | How many times one agent's run may refill the package cache after the change touched a dependency manifest. The broker counts them, so code in the container cannot raise it. `0` turns refills off; a change that adds dependencies may then fail its tests | `3` |
| `WARM_TIMEOUT_SEC` | The time limit (seconds) for one container that fills the cache, which is also the longest wait for another refill to finish. `0` means no limit | `0` |
| `TOOL_OFFLINE` | Whether build tools work from the package cache only. `sandbox` sets it inside the container; do not set it yourself | `false` |

Refills happen only in runs that go through a broker, not in `edit` on the host ([Fetching the dependencies ahead](container.md#fetching-the-dependencies-ahead-warm)). For requests you do not trust, set `WARM_MAX_REFILL=0`.

### Repositories whose tests need another service (`SANDBOX_NETWORK`)

Before the run, create a network that cannot reach outside and run the service on it. The example uses PostgreSQL; docker takes the same arguments.

```bash
# 1. Make a network that cannot reach outside (internal)
podman network create --internal test-net

# 2. Run the service for the tests on that network
podman run -d --name db --network test-net -e POSTGRES_PASSWORD=test postgres:16

# 3. Set the network's name and run
SANDBOX_NETWORK=test-net codetwine-loom sandbox \
  --project ../myapp --output-dir ./out --request "Reject duplicates in user sign-up"
```

The tests reach the service by its container name (`db` above). The host's environment variables are not passed to containers, so put the connection details in the test command: `TEST_COMMAND="env DATABASE_URL=postgresql://postgres:test@db:5432/postgres uv run --with pytest pytest -q"`. The container that fills the package cache does not join this network; the packages are still fetched, but tests run there cannot reach the service. This tool does not start or remove the service's containers.

## The broker and the logs

The broker makes model calls on behalf of containers ([The broker](container.md#the-broker-model-calls)).

| Variable | Description | Default |
|------|------|------------|
| `LLM_BROKER` | Where model calls go. Empty calls the provider from this process. `unix://<path>` or `tcp://<host>:<port>` sends them to a broker; this process then neither reads nor sends `LLM_API_KEY` / `LLM_API_BASE`. `sandbox` sets it inside the container | Empty |
| `LLM_MAX_CALL` | The most model calls one broker lets through. Unlike `RLM_MAX_TURN`, it counts every call, including the planning stage, explorers and retries. `0` means no limit | `0` |
| `BROKER_START_TIMEOUT_SEC` | How long (seconds) `sandbox` waits for its broker to start listening, and to end | `10` |
| `LLM_BROKER_MAX_REQUEST_BYTE` | The most bytes the broker reads as one request. A larger request gets an error and the connection is closed | `8388608` |
| `LLM_BROKER_MAX_CONNECTION` | The most connections one broker serves at once; more are refused | `16` |
| `TURN_LOG_KEEP` | How many runs are kept in `turns/` (one run is a turn log `.jsonl` and an operation log `.log`). The oldest are removed when a new run starts. `0` writes neither | `20` |
| `AGENT_LOG_LEVEL` | The lowest level written to the operation log (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Above `DEBUG`, the terminal shows only progress lines and warnings ([Progress lines](usage.md#progress-lines)). `DEBUG` also prints per-turn lines, token counts and the containers' output | `INFO` |

## Defaults by language

| Variable | Description | Default |
|------|------|------------|
| `FALLBACK_EXT` | The language whose defaults apply when the primary language is not in the table below. Must be an extension in the table | `py` |

The primary language is the one with the most files holding searchable definitions, and within it the extension with the most files. It is recorded in `meta.primary_ext` of `code_index.sqlite`. When some language has a project file at the repository root (a dependency file, Gradle build files for Java, a makefile or `configure` for C / C++, a `.sln` or `.csproj` for C#), Java, Kotlin, JavaScript, TypeScript, C / C++, Rust and C# count only if they have one there too. In an index with no searchable definitions (a repository of SQL queries, say), files with extensions in the table are counted instead. Any other extension uses the defaults of `FALLBACK_EXT`, with a warning. For a change that moves code to another language, the test-writing and refactoring stages use that language instead ([Limitations](limitations.md#tests-of-languages-other-than-the-main-one-run-only-where-that-languages-dependency-file-is)).

| Extension | Test command | Where the tests are | How `paths` are passed |
|--------|---------------|--------------|----------------|
| `py` | `uv run --with pytest --with pytest-asyncio pytest --asyncio-mode=auto -q -W 'ignore:The configuration option "asyncio_default_fixture_loop_scope" is unset.:pytest.PytestDeprecationWarning'` | `tests`, `test`, `testing` | `path` |
| `java` | `mvn -B -fae test` | `src/test/java` | `none` |
| `kt` / `kts` | `gradle test --continue --console=plain` | `src/test/kotlin` | `none` |
| `js` / `jsx` / `mjs` / `cjs` | `npm test --silent` (`node --test` over the test files when the repository has no `package.json`) | `test`, `tests` | `separator` |
| `ts` / `tsx` / `mts` / `cts` | `npm test --silent` | `test`, `tests` | `separator` |
| `c` / `cpp` / `h` / `cc` / `cxx` / `hpp` / `hh` / `hxx` | `ctloom-test-c` | `tests`, `test` | `none` |
| `rs` | `cargo test --workspace --no-fail-fast` | `tests` | `none` |
| `cs` | `dotnet test <solution-at-the-root>` | `tests`, `test` | `none` |
| `sql` | none | `tests` | `none` |
| `cbl` / `cob` / `cpy` | none | `tests` | `none` |
| `bms` | none | `tests` | `none` |
| `jcl` / `prc` / `proc` | none | `tests` | `none` |
| `csd` | none | `tests` | `none` |
| `r` / `rmd` / `qmd` | `ctloom-test-r` | `tests/testthat` | `path` |

- `path`: `TEST_COMMAND` followed by the paths; with no paths, by `TEST_PATH`
- `separator`: `TEST_COMMAND`, then `--`, then the paths; with no paths, `TEST_COMMAND` alone
- `none`: `TEST_COMMAND` alone. Paths are dropped, and the first line of the output says so

For Python, in a directory whose `pyproject.toml` has no `[project]` table (or that has none), `uv run` in the test command and in `RUN_COMMAND` also gets `--isolated`, `--with-editable .` for an installable project, and `--with-requirements` for each usual `requirements*.txt` file. A `setup.py` package built by setuptools whose `setup.py` imports something its build cannot load (Cython, for example) is not installed: `--with-editable .` and the requirement files that install the project itself are left out, and a warning names the package.

By default (`INCLUDE_CONSTANT=false`), search step 2 leaves out variables, constants, fields and enumerators, Rust's modules and associated types, C++'s namespaces, and C#'s properties and indexers. Functions, types, function-like macros and functions assigned to members (`exports.name = ...`) are always included.

Each language's tools write the files below into the target repository (in `.gitignore` syntax; an entry starting with `/` applies only in directories that hold the language's dependency file). In a container's clone they never count as changes, go into the analysis or end up in the patch, even when the repository's `.gitignore` does not list them. Changes to tracked files still count. `grep_source` skips these directories too.

| Extension | Not counted as changes |
|--------|------------------|
| `py` | `__pycache__/`, `uv.lock`, `*.egg-info/`, `/dist/`, `/build/` |
| `java` | `target/` |
| `kt` / `kts` | `build/`, `.gradle/`, `.kotlin/` |
| `js` / `jsx` / `mjs` / `cjs` / `ts` / `tsx` / `mts` / `cts` | `node_modules/`, `.yarn/install-state.gz`, `/dist/`, `/build/`, `/out/`, `/coverage/`, `/.output/`, `.next/`, `.nuxt/`, `.svelte-kit/`, `.angular/`, `.turbo/`, `.parcel-cache/`, `.docusaurus/`, `.astro/`, `*.tsbuildinfo` |
| `rs` | `target/` |
| `cs` | `bin/`, `obj/` |
| `r` / `rmd` / `qmd` | `testthat-problems.rds` |
| others | none |

For COBOL, only records (top-level group items) are search targets; the names of the other data items are added as words of their record, so searching for an item finds its record. For BMS screen definitions, the symbolic maps (`<map>I`, `<map>O`) are the records and their fields the words. COBOL, BMS, JCL and CSD names are not case-sensitive. When a repository has COBOL files, BMS, JCL and CSD files count as COBOL when the primary language is chosen. Files that codetwine analysed as a different language than their extension suggests (copybooks named `.dcl`, `.inc` or with no extension, files recognised as JCL by their content) are handled as the analysed language.
