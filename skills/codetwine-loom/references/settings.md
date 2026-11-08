# Settings

Every setting is an environment variable with exactly the name below. A value comes from the first of: a command-line argument, the environment, `.env` in the current directory, the `PRESET` column, the default. An empty value (`NAME=`) means the default for most settings. A value a setting cannot take stops the command at start-up with one line naming it. Containers never read `.env`; the tool passes the settings to them by name.

This page lists the settings an agent is likely to need. The tool's `docs/settings.md` lists every one.

## Contents

- [Models and keys](#models-and-keys)
- [`PRESET`](#preset)
- [Tests, static check and running the program](#tests-static-check-and-running-the-program)
- [Cost and time limits](#cost-and-time-limits)
- [Containers](#containers)
- [The stages after the merge](#the-stages-after-the-merge)
- [Search and analysis](#search-and-analysis)
- [Web search and fetch](#web-search-and-fetch)
- [Proxies](#proxies)
- [Logs and the report](#logs-and-the-report)

## Models and keys

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | The API key sent with every model call. Serves one provider | empty |
| `<PROVIDER>_API_KEY` | With models of two or more providers, leave `LLM_API_KEY` empty and set each provider's own variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). Bedrock also accepts `AWS_BEARER_TOKEN_BEDROCK` or `AWS_PROFILE` | — |
| `MEDIUM_MODEL` | The model of the stages that do the work (litellm form `provider/model`). `EDIT_MODEL`, `ANSWER_MODEL`, `EXPLORE_MODEL`, `QUERY_EXPAND_MODEL`, `REPORT_MODEL`, `TEST_CREATE_MODEL`, `REFACTOR_MODEL`, `WEB_SEARCH_MODEL` and `DOC_CREATE_MODEL` follow it | `anthropic/claude-sonnet-5-5` |
| `HIGH_MODEL` | The model of the stages that decide what to do. `PLAN_MODEL`, `DISPATCH_MODEL` and `INTEGRATE_MODEL` follow it | `anthropic/claude-opus-5-5` |
| `LLM_API_BASE` | A non-standard endpoint (a local model server and the like). It must not answer with redirects | empty |
| `THINK_LEVEL` | How hard one call thinks: `off`, `low`, `medium`, `high`. The other stages' `*_THINK_LEVEL` follow it | `PRESET` column |
| `AGENT_MAX_TOKEN` | The output limit of one call, thinking and reply together. The other stages' `*_MAX_TOKEN` follow it | `PRESET` column |
| `LLM_CALL_TIMEOUT_SEC` | The time limit of one attempt of one call | `600` |
| `HISTORY_MAX_CHAR` | The most characters of history put into a prompt. Lower it for models with a small context | `600000` |

Each per-stage model (`EDIT_MODEL`, `PLAN_MODEL`, …) can be set on its own when the user wants one stage on another model.

## `PRESET`

`low` / `medium` / `high` (default `medium`; `--preset` for one run). It picks a column for these settings; a setting you set yourself wins. It does not set the models or the turn limits.

| Variable | `low` | `medium` | `high` | What changes |
|---|---|---|---|---|
| `THINK_LEVEL` | `off` | `low` | `medium` | How hard one call thinks |
| `AGENT_MAX_TOKEN` | `16384` | `32768` | `32768` | The output limit of one call |
| `EXPLORE_MAX_CALL` | `3` | `4` | `5` | `explore` calls per change run |
| `SEARCH_AGAIN_MAX_CALL` | `3` | `5` | `10` | `search_again` calls |
| `WEB_SEARCH_MAX_CALL` | `10` | `15` | `20` | `web_search` calls |
| `WEB_FETCH_MAX_CALL` | `5` | `8` | `10` | `web_fetch` calls |
| `RUN_PROJECT_MAX_CALL` | `10` | `15` | `20` | `run_project` calls per stage |
| `MAX_PATCH_ROUND` | `3` | `5` | `8` | Rounds of "rewrite and test" |
| `TWO_STAGE` | `false` | `true` | `true` | A read-only planning stage before the change |
| `DISPATCH_MAX_TASK` | `5` | `8` | `10` | Pieces `run` splits one request into |
| `DISPATCH_MAX_RETRY` | `2` | `3` | `5` | Hand-outs again of a task that did not finish |
| `DISPATCH_MAX_RERUN` | `1` | `2` | `3` | Reruns of a request that did not succeed |
| `INTEGRATE_MAX_REFUSAL` | `1` | `2` | `3` | Send-backs of the merge's finish while machine checks remain |
| `INTEGRATE_MAX_RERUN` | `0` | `1` | `2` | Redos of an unsettled merge in `run` |
| `TEST_CREATE_MAX_RERUN` | `1` | `2` | `3` | Redos of an unsettled test-writing stage |
| `REFACTOR_MAX_RERUN` | `1` | `2` | `3` | Redos of an unsettled refactoring stage |

## Tests, static check and running the program

The defaults come from the repository's primary language (the one `build` prints as `primary_ext:`). Set these when the defaults do not fit the repository.

| Variable | Description | Default |
|---|---|---|
| `TEST_COMMAND` | The test command, split like a shell command line. When set, only it runs | The primary language's default plus the defaults of other languages whose dependency file is present |
| `TEST_PATH` | Where the tests are, relative to the repository root; also where the first tests go in a repository with none | The primary language's test directory with the most files |
| `TEST_TARGET_STYLE` | How test paths are passed: `path` (appended), `separator` (after `--`), `none` (dropped). Set it with `TEST_COMMAND` when you switch build tools | The language's default |
| `TEST_TIMEOUT_SEC` | The time limit of one test run | `300` |
| `CHECK_COMMAND` | The static check. Each output line `<path>:<line>[:<col>]: <text>` is a finding; new findings leave the change unsettled | The language's default (`ruff check --select F821` for Python; none for Java, Kotlin, SQL) |
| `RUN_COMMAND` | The start of the command `run_project` runs; the model passes the rest | `uv run` (Python), `cargo run --quiet --` (Rust), `Rscript` (R); none for most others |
| `SMOKE_ARG` | Arguments after `RUN_COMMAND` for a start of the program after every turn that changed files. `{workspace}` is a temporary directory. Empty means no check | empty |
| `FORMAT_COMMAND` | A formatter run in the container on the changed files at the end of each run (`ruff format`). Must exist in the image | empty |
| `COBOL_DIALECT` | `cobc -std=` dialect for the COBOL check (`ibm`, `mf`, …) | empty |

Default test commands by primary language:

| Language | Test command | Tests in |
|---|---|---|
| Python | `uv run --with pytest --with pytest-asyncio pytest ... -q` | `tests`, `test`, `testing` |
| Java | `mvn -B -fae test` | `src/test/java` |
| Kotlin | `gradle test --continue --console=plain` | `src/test/kotlin` |
| JavaScript / TypeScript | `npm test --silent` (`node --test` without `package.json` for JavaScript) | `test`, `tests` |
| C / C++ | `ctloom-test-c` (needs CMake or a makefile `test` target) | `tests`, `test` |
| Rust | `cargo test --workspace --no-fail-fast` | `tests` |
| C# | `dotnet test <solution-at-the-root>` | `tests`, `test` |
| R | `ctloom-test-r` (testthat) | `tests/testthat` |
| SQL, COBOL, BMS, JCL, CSD | none; set `TEST_COMMAND` | `tests` |

Tests that need another service (a database): create an internal network (`podman network create --internal test-net`), run the service on it, set `SANDBOX_NETWORK=test-net`, and put the connection details in `TEST_COMMAND` (`env DATABASE_URL=postgresql://...@db:5432/... uv run --with pytest pytest -q`), because the host's environment is not passed to containers.

## Cost and time limits

| Variable | Description | Default |
|---|---|---|
| `MAX_TOKEN_PER_RUN` | Tokens (input and output) one run of `run`, `sandbox` or `integrate` may use across all its agents. When reached, no new container starts and the run exits with 1 | `0` (no limit) |
| `LLM_MAX_CALL` | Model calls one broker (one agent) lets through | `0` (no limit) |
| `DISPATCH_PARALLEL` | Agents `run` runs at once. Limits and memory apply per agent | `3` |
| `RLM_MAX_TURN` | The turn limit of the change stage (the answering, planning and splitting stages follow it) | `40` |
| `INTEGRATE_MAX_TURN`, `TEST_CREATE_MAX_TURN`, `REFACTOR_MAX_TURN` | Turn limits of the merge and the stages after it | `50` |
| `SANDBOX_TIMEOUT_SEC` | The time limit of one agent's container; a container that runs out counts as not finished | `0` (no limit) |
| `DISPATCH_KEEP_RUN` | Resumable runs kept per command; older branches and directories are removed | `3` |

Tokens spent by `--doc-create` are not counted in `MAX_TOKEN_PER_RUN`.

## Containers

| Variable | Description | Default |
|---|---|---|
| `CONTAINER_ENGINE` | `podman` or `docker` (`--engine`) | empty: `podman`, then `docker` |
| `CONTAINER_IMAGE` | The image (`--image`) | `codetwine-loom` |
| `SANDBOX_MEMORY` | Memory per container (`2g`, `512m`). Reaching it ends the container with 137 | `2g` |
| `SANDBOX_MAX_PROCESS` | Processes per container | `4096` |
| `SANDBOX_NETWORK` | An internal container network the run's containers join | empty (no network) |
| `SANDBOX_CACHE_DIR` | Where the package cache lives; set it to share one cache between output directories | `<output-dir>/agents/.repo/<repo-name>/cache` |
| `WARM_MAX_REFILL` | Times one agent may refill the package cache, with a network, after a change edits a dependency file. Refills run the repository's build files, which the change may have written: set `0` for requests the user does not trust | `3` |

## The stages after the merge

| Variable | Description | Default |
|---|---|---|
| `TEST_CREATE` | Write tests for the changed definitions (`--no-test-create` sets `false`) | `true` |
| `REFACTOR` | Refactor the changed implementation files (`--no-refactor`) | `true` |
| `TEST_REFACTOR` | Refactor the changed tests (`--no-test-refactor`) | `true` |
| `TEST_CREATE_MAX_DEFINITION` | About how many changed definitions one test task covers before the work is split | `10` |

These stages end without calling the model when there is no test command or no readable starting test result.

## Search and analysis

| Variable | Description | Default |
|---|---|---|
| `OUTPUT_DIR` | Where the analysis, index and records go (`--output-dir`) | `./out` |
| `BUILD_EXCLUDE` | Names kept out of the analysis, shell-quoted (`"vendor migrations"`). `--exclude` adds to it | empty |
| `OVERVIEW` | Make the overall overview and pass it to search (`--no-overview` sets `false`) | `true` |
| `DOC_CREATE_MODEL` | The model codetwine writes design documents with (`--doc-create` only) | `MEDIUM_MODEL` |
| `DOC_CREATE_EFFORT` | How hard it thinks for design documents: `off`, `low`, `medium`, `high` | `low` |
| `DOC_TEMPLATE_PATH` | The design-document template | `doc_template.json` of the codetwine-loom clone |
| `SOURCE_ENCODING` | Encodings to try for files that are not UTF-8 (`cp1252,iso8859_15`) | empty |
| `CODETWINE_MAX_WORKER` | Files codetwine analyses in parallel | `4` |

## Web search and fetch

Without a provider, the agents get no web tools. In container runs only the broker on the host uses these keys.

| Variable | Description | Default |
|---|---|---|
| `WEB_SEARCH_PROVIDER` | `model` (the model provider's own search, with the LLM key: Anthropic, OpenAI or Gemini), `tavily`, `brave` or `ollama` | empty (no `web_search`) |
| `WEB_SEARCH_API_KEY` | The search API's key (not needed for `model`) | empty |
| `WEB_FETCH_PROVIDER` | `tavily`, `ollama` or `jina`. Reads only URLs `web_search` returned; needs a search provider | empty (no `web_fetch`) |
| `WEB_FETCH_API_KEY` | The fetch API's key. Defaults to `WEB_SEARCH_API_KEY` when both providers are the same | empty |

## Proxies

| Variable | Description |
|---|---|
| `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` | `http://` or `https://` proxy URL. User and password in ASCII; percent-encode `#`, `/`, `?`, `[`, `]`, `\` |
| `NO_PROXY` | Hosts reached directly, comma-separated, host only (no scheme or port) |
| `SSL_CERT_FILE` | A PEM file of certificate authorities, for a proxy that signs TLS itself. Also passed to the image build |

The containers that fetch dependencies get these; the containers that make changes do not. A proxy on `127.0.0.1` cannot be reached from containers.

## Logs and the report

| Variable | Description | Default |
|---|---|---|
| `AGENT_LOG_LEVEL` | `DEBUG` prints every line including per-turn lines; `WARNING` only warnings | `INFO` |
| `TURN_LOG_KEEP` | Runs kept in each `turns/` directory | `20` |
| `REPORT_MODEL` | The model that writes `run`'s closing report. Empty writes only the file list | `MEDIUM_MODEL` |
| `EDIT_GUIDE`, `TEST_GUIDE`, `REFACTOR_GUIDE`, `TEST_REFACTOR_GUIDE` | The instruction files (see [instruction-files.md](instruction-files.md)) | `CODETWINE_LOOM*.md` |
