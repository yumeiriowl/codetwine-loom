# Containers

This page expands the README's [Running in containers](../README.md#-running-in-containers). `sandbox`, `warm`, and `run` and `integrate` (without `--here`) start containers. Their settings are in [settings.md](settings.md#sandbox), and the files they leave are in [output.md](output.md#files).

## Contents

- [Building the image](#building-the-image)
- [The boundary `sandbox` sets up](#the-boundary-sandbox-sets-up)
- [The broker (model calls)](#the-broker-model-calls)
- [Fetching the dependencies ahead (`warm`)](#fetching-the-dependencies-ahead-warm)
- [How a result reaches the working tree](#how-a-result-reaches-the-working-tree)
- [What a run that did not pass leaves](#what-a-run-that-did-not-pass-leaves)
- [Carrying on to a second round](#carrying-on-to-a-second-round)
- [Running several agents](#running-several-agents)
- [Combining several branches into one (`integrate`)](#combining-several-branches-into-one-integrate)
- [Before a change goes into the working tree](#before-a-change-goes-into-the-working-tree)

## Building the image

Every container of a run uses one image. Unless `--image` or `CONTAINER_IMAGE` names another, it is `codetwine-loom`, which holds the toolchains of every language.

You do not have to build it first. Before starting, the subcommands that start containers, and `setup`, check the image ([Checking the keys and the image at start-up](settings.md#checking-the-keys-and-the-image-at-start-up)). An image this tool builds carries a hash of the files it is built from (`Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`, `uv.toml`, `codetwine_loom/` and `sandbox/`) in the label `io.codetwine-loom.source`, and its `TOOLCHAIN` in the label `io.codetwine-loom.toolchain`. The image is *missing* when the engine does not have it, and *stale* when that hash is absent or differs from the installed files, for example after you upgrade codetwine-loom or codetwine ([Following codetwine](usage.md#following-codetwine)).

At a terminal, the tool prints the image, its `TOOLCHAIN` and the build command, and asks `Build it now? [Y/n]` (`Build it again now? [Y/n]` for a stale image; `setup` asks with `[y/N]`). On yes, it builds the image and the command goes on. On no, or away from a terminal, a missing image stops the command with the build command, and a stale image is used with a warning. A build that fails or takes more than an hour stops the command. An image named by `--image` or `CONTAINER_IMAGE` is rebuilt with the `TOOLCHAIN` its label names, or with every toolchain when it has no label. When codetwine-loom is not installed from a clone of its repository, there is no `Dockerfile` to build from: a missing image stops the command, and an existing image is used without checking it.

Behind a proxy, the tool passes the proxy variables that have a value to `docker build` as `--build-arg` (`podman build` passes them itself). The certificates in `SSL_CERT_FILE` that Mozilla's bundle does not hold go in as the build argument `CA_PEM`. When they come to more than 100,000 bytes, nothing is built and the command stops; point `SSL_CERT_FILE` at a file of only the authorities you add ([Proxies and certificates](settings.md#proxies-and-certificates)).

To build by hand, use the command the check prints. An image built without its `--label io.codetwine-loom.source=<hash>` counts as stale, so the commands below make images the check warns about (or offers to rebuild at a terminal). An image with fewer toolchains is smaller, but the languages it lacks cannot be built, tested or checked in it (`run_test()` returns `"did not run"`).

```bash
# Build the image with every toolchain
podman build -t codetwine-loom .
# Build an image with the toolchains of some languages only (separate them with spaces or commas)
podman build --build-arg TOOLCHAIN="java ts" -t codetwine-loom-java-ts .
# Build behind a proxy that signs TLS with its own certificate authority (pass the PEM of the proxy's CA)
podman build --build-arg CA_PEM="$(cat corp-ca.pem)" -t codetwine-loom .
docker build --build-arg CA_PEM="$(cat corp-ca.pem)" --build-arg HTTPS_PROXY="$HTTPS_PROXY" -t codetwine-loom .
# Install two .NET SDK channels
podman build --build-arg DOTNET_CHANNEL="10.0 8.0" -t codetwine-loom .
```

`TOOLCHAIN` takes the values in the first column below; the default is `c cbl cs java js kt r rs ts`. `py` and `sql` are accepted and add nothing, and any other value stops the build. Every image holds Python 3.12, git and `uv`. The tool's own dependencies are installed at the versions in `uv.lock`, and the build fails when `pyproject.toml` or `uv.toml` disagrees with it. The base images (`python:3.12-slim`, `ghcr.io/astral-sh/uv:latest`) are taken from the host when it has them, and the uv in the image must be 0.11.8 or later. A rebuild with the same arguments reuses cached steps. To renew the base images and packages, build with `podman build --no-cache --pull=newer` (`docker build --no-cache --pull`).

| `TOOLCHAIN` | What it installs | Commands it adds |
|---|---|---|
| `java` | Debian's `default-jdk` and `maven`, and Temurin JDK 11 (`JDK11_VERSION`; amd64 and arm64). Maven uses JDK 11 for builds whose `pom.xml` files target Java 8 or older and none targets above 11 | — |
| `kt` / `kts` | `default-jdk` and Gradle (`GRADLE_VERSION`, default `8.14.6`) | `gradle` |
| `js` / `ts` (and `jsx`, `mjs`, `cjs`, `tsx`, `mts`, `cts`) | Debian's `nodejs` and `npm`; `g++` and `make` for native packages; `pnpm` and `yarn` through corepack (`COREPACK_VERSION`); Bun (`BUN_VERSION`, default `1.4.4`; x86-64 and ARM64); ESLint 10; TypeScript 5 with the type definitions of Node.js, jest and Bun. No browser, so browser tests (karma and the like) do not run | `bun`, `bunx`, `ctloom-check-js` (`js`), `ctloom-check-ts` (`ts`), `ctloom-install-js` |
| `c` / `cpp` / `h` / `cc` / `cxx` / `hpp` / `hh` / `hxx` | GCC (`build-essential`), CMake, Ninja, autotools, `pkgconf`, bison and flex | `ctloom-check-c`, `ctloom-test-c`, `ctloom-run-c` |
| `rs` | Rust (`RUST_VERSION`, default `1.99.1`) with GCC, `make`, `pkg-config` and `libssl-dev`. `rust-toolchain.toml` is not read | `ctloom-check-rs` |
| `cs` | The .NET SDK (`DOTNET_CHANNEL`, several separated by spaces; default `10.0`). Add a version here when `global.json` asks for one the image lacks | `ctloom-check-cs` |
| `cbl` (and `cob`, `cpy`, `bms`, `jcl`, `prc`, `proc`, `csd`) | GnuCOBOL | `ctloom-check-cbl` |
| `r` (and `rmd`, `qmd`) | R, testthat and codetools. `renv.lock` is not read | `ctloom-check-r`, `ctloom-test-r` |

The `ctloom-check-*` commands are the default static checks (`CHECK_COMMAND`) of their languages. They run at the top of the target repository and rewrite none of its files; what each finds is in [The default static checks differ in strength by language](limitations.md#the-default-static-checks-differ-in-strength-by-language). `ctloom-test-c` and `ctloom-test-r` are the default test commands for C / C++ and R ([C / C++](limitations.md#c--c-tests-need-cmake-or-a-makefile-with-a-test-target), [R](limitations.md#r-tests-are-testthats-teststestthat-only)). `ctloom-run-c` is the default launcher of `run_project` for a C / C++ repository with a makefile: `run_project(["demo"])` runs `make demo`. The C / C++ commands build in a copy of the repository or a separate build directory, so they write nothing to it.

`ctloom-install-js` installs the packages `package.json` names into `node_modules`. Inside a container it runs before each JavaScript or TypeScript test run, static check and `run_project` start; on the host it does not run, and the packages already in `node_modules` are used. It uses the package manager the `packageManager` field names, or else the one whose lock file is there (npm, then pnpm, then yarn, then Bun, which `bunfig.toml` also selects), or else npm. It does nothing when the lock files and `package.json` files are unchanged since the last install. The lock file is left as it was, unless the change altered the dependencies so that an install holding to the lock file failed; then the lock file is regenerated and becomes part of the change. When the install fails, no test runs and the result is `"did not run"`. `node_modules` is not counted among the changed files.

## The boundary `sandbox` sets up

Before a run, the host makes a commit that copies the working tree as it is, as a child of `HEAD` (`HEAD` itself when the working tree is clean). This commit is the run's *base*. It holds uncommitted changes and untracked files, but not what `.gitignore` ignores and not the contents of submodules ([The base is the working tree itself](limitations.md#the-base-is-the-working-tree-itself)). `HEAD`, the index and the working tree do not move.

A container that makes a change gets these mounts and nothing else:

| Inside | What it is |
|---|---|
| `/src` (read-only) | A bare copy of the target repository's branches, tags and objects. The working tree, ignored files, `.git/config` and hooks are not in it. The container clones it and checks out the base |
| `/work`, `/tmp` | The clone and `run_project`'s working area, on the host's disk. They are made for each run and removed when it ends |
| `/cache` | The toolchains' package cache, also `HOME` inside the container. It is shared by every container and kept across runs. Default `<output-dir>/agents/.repo/<repo-name>/cache` (`SANDBOX_CACHE_DIR`) |
| `/run/ctloom` | The broker's socket, the only way out |
| `/out` | That agent's place, `<output-dir>/agents/<agent>/`: its copy of the index, its records and its patch. It is the only written place that remains after the run. No container gets the output directory itself |

The container runs with `--network=none` (or on the internal network `SANDBOX_NETWORK` names; [Repositories whose tests need another service](settings.md#repositories-whose-tests-need-another-service-sandbox_network)) and without the host's proxy variables. Its image file system is read-only, it has no privileges (`--cap-drop=ALL`, `--security-opt=no-new-privileges`), and the number of processes and the memory are limited (`SANDBOX_MAX_PROCESS`, `SANDBOX_MEMORY`). The clone and the cache are on the host's disk and do not count toward the memory limit, and how much is written is not limited. Files written to `/out` are owned by the user who ran the command. What only the host writes (the diff applied to the working tree, the container's output and the broker's log) goes to `<output-dir>/agents/.host/<agent>/`, which no container sees.

Every setting is passed to the container by name, except these:

| Settings not passed | How they are handled |
|---|---|
| `PROJECT_DIR`, `OUTPUT_DIR`, `SANDBOX_CACHE_DIR` | Paths on the host. The container gets its own paths |
| `BUILD_EXCLUDE`, `LLM_BROKER`, `TOOL_OFFLINE`, `WEB_SEARCH_MAX_CALL`, `WEB_FETCH_MAX_CALL`, `EDIT_GUIDE`, `EDIT_HISTORY` | The host sets the value. `TOOL_OFFLINE` is `true` in a container without a network, the two web limits are `0` when the host has no search or fetch provider, and `EDIT_HISTORY` is `true` |
| `LLM_API_KEY`, `LLM_API_BASE`, `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_API_KEY`, `WEB_SEARCH_MODEL`, `WEB_SEARCH_MAX_TOKEN`, `WEB_SEARCH_CONTEXT_SIZE`, `WEB_SEARCH_TIMEOUT_SEC`, `WEB_SEARCH_MIN_INTERVAL_SEC`, `WEB_FETCH_PROVIDER`, `WEB_FETCH_API_KEY`, `WEB_FETCH_TIMEOUT_SEC`, `WEB_FETCH_MIN_INTERVAL_SEC` | Only the broker on the host reads them. The container sends only the question or the URL |

The keys are also kept from the programs the tool starts, on the host and in containers ([Programs that are started get no keys](limitations.md#programs-that-are-started-get-no-keys)). The container cannot send to a destination of its choice, but what it reads can still reach the configured provider inside a model call ([What `sandbox` holds down is the destination, not the contents](limitations.md#what-sandbox-holds-down-is-the-destination-not-the-contents)).

## The broker (model calls)

A container without a network cannot reach a provider. `sandbox` starts a broker on the host for each change, and the container sends its requests over the socket as JSON-RPC, one message per line. The broker serves 4 methods: `completion` (a model call), `search` (`web_search`), `fetch` (`web_fetch`) and `refill` (refill the package cache).

```text
container                         host
  make_llm()                       broker.serve()
    └─ JSON-RPC ─→ /run/ctloom/llm.sock ─→ adds api_key, on to litellm
    ←──── only the reply text, finish_reason and token counts ────┘
```

The broker adds `LLM_API_KEY` and `LLM_API_BASE` itself and refuses a call that carries them. It also refuses a `completion` whose model is not the value of a `*_MODEL` setting, whose `max_tokens` is above the largest `*_MAX_TOKEN` setting, whose thinking settings are not a form `THINK_LEVEL` makes, or whose messages are not text only. A refused call is not sent and not counted; inside the container it arrives as a failed call. Only the reply text, `finish_reason` and the token counts come back. A reply longer than `max_tokens` × 32 characters is cut, with `finish_reason` `length`. The keys and the proxy values are removed from failure messages and from the broker's log (only values of 8 characters or more; the broker warns about a shorter key). Redirects from the endpoint are not followed ([Write an endpoint that does not return redirects](limitations.md#write-an-endpoint-that-does-not-return-redirects-in-llm_api_base)).

The broker enforces the call limits; the counter inside the container only tells the model how many calls remain. It also refuses model calls once the run reaches `MAX_TOKEN_PER_RUN` ([Cost limits](limitations.md#cost-limits)).

| Method | Limit | Default |
|---|---|---|
| `completion` | `LLM_MAX_CALL` | `0` = no limit |
| `search` | `WEB_SEARCH_MAX_CALL` | `PRESET` column (`15` for medium) |
| `fetch` | `WEB_FETCH_MAX_CALL` | `PRESET` column (`8` for medium) |
| `refill` | `WARM_MAX_REFILL` | `3` |

The limits count per broker process. `sandbox` starts and stops a broker for each change, so they apply per run, and per agent in `run`. A broker you keep running counts across runs. To run one on its own: `codetwine-loom broker --endpoint unix:///path/to/llm.sock --project <target-repo> --output-dir <output-dir>`. Without `--project` it serves model calls only and refuses refills. A `tcp://` broker has no authentication ([A `tcp://` broker has no authentication](limitations.md#a-tcp-broker-has-no-authentication)). The forms of `LLM_BROKER` are in [The broker and the logs](settings.md#the-broker-and-the-logs).

## Fetching the dependencies ahead (`warm`)

In a container without a network, `uv`, `mvn`, `gradle`, `npm` and the like cannot reach their registries. They work from the package cache (`/cache`) alone. To fill it, a container with a network and without keys runs the test command once over a clone of the base. This container gets the host's proxy settings ([Proxies and certificates](settings.md#proxies-and-certificates)).

You do not have to run `warm` first. Before `sandbox`, `run` or `integrate` start a container, they check which dependency manifests and image the cache has been filled for. When the base's manifests or the image are new to it (after a rebuild of the image, or after a commit added dependencies), the fill runs once. A fill also runs the tests of another language when the base adds files of a language with a default test command. A fill that failed, or that stopped before the tests because something could not be fetched, is not recorded. Later containers of the same command try again until it has failed twice, and the next command tries again. When no test command would run, nothing is filled. `warm` fills the cache again by hand, even when it is already filled for the base:

```bash
codetwine-loom warm --project /path/to/target-repo --output-dir ./out
```

When a change touches a dependency manifest (`pyproject.toml`, `pom.xml`, `package.json`, `build.gradle` and the others of each language), `run_test()`, `run_project` and the automatic check ask the broker for a *refill* before they start. The broker runs the same fill in a container with a network and no keys, over the base with only the change's manifest files laid over it. Other code the change wrote does not reach it, and the model sees no added tool or instruction. When the cache is already filled for those manifests, nothing starts and nothing is counted. Every refill that starts counts against `WARM_MAX_REFILL`, even one that fails. After a failed refill the tests run anyway, with a `note:` line at the head of `run_test()`'s output, and what is missing shows up as a failure. Once the count is used up, code that needs new dependencies fails.

> **⚠️ The manifests are code.** `npm test` runs `scripts.test` of `package.json`, `gradle test` evaluates `build.gradle(.kts)`, `pom.xml` can bind any plugin to the test phase, and `setup.py` can run while dependencies are resolved. In a refill, the model wrote those manifests, and the container has a network. **When running requests you do not trust, set `WARM_MAX_REFILL=0`**; changes that add dependencies then do not pass their tests. See [`warm` and refills use the network](limitations.md#warm-and-refills-use-the-network) and [The package cache is writable by every container](limitations.md#the-package-cache-is-writable-by-every-container-and-containers-with-a-network-read-it-too).

## How a result reaches the working tree

When the change is done, the container writes one patch: the difference from the base, including anything committed inside the clone. Paths git could not add (unreadable files, nested repositories without commits) are left out and listed in the `.meta.json` beside the patch. An empty patch means no change.

The host commits the patch to a branch without touching the working tree. The commits are authored by `codetwine-loom <codetwine-loom@localhost>` and are not signed. When the run succeeded (exit status 0), the change goes through the checks in [Before a change goes into the working tree](#before-a-change-goes-into-the-working-tree), is applied to the working tree, and the branch is removed. The index is not touched, so the change shows in `git status` and `git diff`, and you make the commit. When the run did not succeed, nothing is applied and the branch stays ([next section](#what-a-run-that-did-not-pass-leaves)).

When nothing could be committed (the container ended before writing the patch, the patch was over the size limit, or a submodule directory holds files its commit lacks), the last line of `sandbox` is `NOT taken in: <reason>`, and a `run` task says `it did not finish: <reason>`. When a run that ended with a non-zero status left an empty patch, the last line is `NOT applied to the working tree: <reason>` instead of `changed: nothing`.

The commit's subject is a prefix (`sandbox:` for a change, `integrate:` for a merge, the stage's name for a stage after the merge) followed by the request cut at 72 characters, its body is the request, and its last paragraph holds git trailers that record what the tool decided:

```text
sandbox: Keep loading the settings file from crashing

Keep loading the settings file from crashing

Codetwine-Loom-Run-Request: Keep loading the settings file from crashing
Codetwine-Loom-Attempt: 2
Codetwine-Loom-Verdict: unverified
Codetwine-Loom-Why: nothing here checks that the request itself is done, so the work may still be part-way
Codetwine-Loom-Exit-Status: 0
```

| Trailer | Contents |
|---|---|
| `Codetwine-Loom-Run-Request` | The request the run received, in one line. In a `run` task's commit the body is the task text, and the original request is only here. `run --resume` and `integrate --resume` read it as the request |
| `Codetwine-Loom-Verdict` | For a change, `failed`, `unverified`, `no_change` or `tests_red`; for a merge and the stages after it, `settled` or `unsettled` |
| `Codetwine-Loom-Why` | The reason for the verdict |
| `Codetwine-Loom-Did` | What the run said it did, when it said anything |
| `Codetwine-Loom-Attempt` / `Codetwine-Loom-Exit-Status` | Which attempt the change is, and the exit status inside the container |
| `Codetwine-Loom-Tests-Changed` / `Codetwine-Loom-Untooled` | The test files the change edited, and the files written without the tools ([The REPL is not a sandbox](limitations.md#the-repl-is-not-a-sandbox)) |
| `Codetwine-Loom-Stage` | Only on the commits of the stages after the merge: `test-create`, `refactor` or `test-refactor` |

Other `Codetwine-Loom-*` lines hold state that `run --resume` and `integrate --resume` read back, such as what a merge decided. Only what the tool decided goes in; to add your own notes, use `git notes`. `merge`, `rebase` and `cherry-pick -x` keep the trailers; `git merge --squash` loses them. git can print them:

```bash
git log --format='%h attempt=%(trailers:key=Codetwine-Loom-Attempt,valueonly,separator=%x20) verdict=%(trailers:key=Codetwine-Loom-Verdict,valueonly,separator=%x20)' <branch>
```

## What a run that did not pass leaves

A run whose exit status is not 0 (it used up its turns, crashed, or ended with the checks or the program start still failing) is not applied. Its work stays on the branch `codetwine-loom/sandbox/<run-id>`, and the last line gives the command to continue it: `to carry on from what this run left: run --resume codetwine-loom/sandbox/<run-id>`. An unsettled merge of `run` or `integrate` stays on a branch of the same form ([Combining several branches into one](#combining-several-branches-into-one-integrate)).

To read the work, and to apply it once you have decided to (`<base>` is the `base:` line of the output):

```bash
git -C <repo> diff --no-textconv --no-ext-diff <base> codetwine-loom/sandbox/<run-id>
git -C <repo> diff --binary --no-textconv --no-ext-diff --no-color --src-prefix=a/ --dst-prefix=b/ \
    <base> codetwine-loom/sandbox/<run-id> | git -C <repo> apply --binary --whitespace=nowarn --no-ignore-whitespace
```

A `git apply` by hand skips the checks in [Before a change goes into the working tree](#before-a-change-goes-into-the-working-tree). Before applying, check whether the diff touches settings files (`.env` and others), hooks, places git reads as configuration, or paths git ignores. `git apply` does not move a submodule's commit.

## Carrying on to a second round

Give the request again. The first round's result is in the working tree, and the second round's base copies the working tree with it, whether you committed it or not.

```bash
codetwine-loom sandbox --project <repo> --output-dir ./out --request "<request>"
git -C <repo> diff          # Check the applied changes. Commit whenever you like
codetwine-loom sandbox --project <repo> --output-dir ./out --request "<request>"
```

When the analysis does not match the new base, the container brings its own copy of the index up to date without design documents and without calling a model ([Automatic rebuilds make no design documents](limitations.md#automatic-rebuilds-make-no-design-documents)). The index under `OUTPUT_DIR` is not rewritten. When the index was built with `--exclude`, `sandbox` uses the recorded names; given different names, it stops and shows the `build --exclude ...` command to run.

Only the code carries over: a change inside a container always starts fresh and does not know what the previous run did. To tell the next run what an earlier run left, use `run --resume`, which reads it from the branch's commits ([Carrying on from an earlier run](usage.md#carrying-on-from-an-earlier-run-run---resume)).

## Running several agents

Running agents side by side is the job of [`run`](usage.md#how-run-handles-a-request): it splits the request, starts the tasks at the same time, merges their branches with `integrate`, and applies the union to the working tree. Do not run several `sandbox` commands at once on the same working tree; the one that ends later may not apply. One `--output-dir` is enough.

```bash
REPO=/path/to/target-repo
codetwine-loom build --project "$REPO" --output-dir ./out              # the index, once
codetwine-loom warm  --project "$REPO" --output-dir ./out              # optional; the cache is shared
codetwine-loom run   --project "$REPO" --output-dir ./out --request "..."
git -C "$REPO" status && git -C "$REPO" diff                           # review, then commit
```

Each agent has one broker, one container and its own place `<output-dir>/agents/<agent>/` (a task of `run` is named `codetwine-loom-run-<run-id>-<number>`). At the start, the agent copies the analysis and index that `build` made into its place, so a rebuild for a moved base changes only that copy. The copy is removed at the end; the patch, the turn log and the records remain. The package cache is shared. `LLM_MAX_CALL`, `WEB_SEARCH_MAX_CALL`, `WEB_FETCH_MAX_CALL`, `WARM_MAX_REFILL` and `SANDBOX_MEMORY` apply to each agent, so N agents can use N times as much, and the host needs N times the memory.

The containers are named `ctloom-<run-id>` (visible in `podman ps`). `run`'s tasks commit to `codetwine-loom/run/<run-id>/<number>`, and the union goes to `codetwine-loom/sandbox/<run-id>`; the branches are removed once applied. When the target repository has a branch named `codetwine-loom`, `codetwine-loom/run` or `codetwine-loom/sandbox`, the commands that start containers stop before starting.

## Combining several branches into one (`integrate`)

Branches changed separately from the same base have never seen each other's results, and their union can break in two ways:

| How it breaks | Example | git |
|---|---|---|
| **Text overlaps** | Two branches touch the same line | Stops it as a conflict |
| **A reference breaks** | A renames a function, and B adds a call to it (no lines overlap) | Lets it through silently |

`integrate` builds the union in a container and checks it by machine before applying it: no conflict markers remain, the conflicted files have been read, the merge broke no references, the tests are no worse than on the merge target, the static check gives no new findings, and the definitions each branch added or changed are in the union ([7. Merging branches](processing-flow.md#7-merging-branches-integrate)). When all checks pass and no definition that two or more branches reached is left to read, no model is called. The current working tree, uncommitted changes included, is laid on first as one of the sides, so only what the union adds is applied. `run` passes its task branches here itself. To combine branches people made, give two or more with `--branch`; they are merged in the order given.

```bash
codetwine-loom integrate \
  --project "$REPO" --output-dir ./out \
  --branch codetwine-loom/agent1 --branch codetwine-loom/agent2 --branch codetwine-loom/agent3
```

```text
merged: codetwine-loom/run/20260903-215341-1-31577/tree
merged: codetwine-loom/agent1
merged: codetwine-loom/agent2 (2 conflicted file(s))
merged: codetwine-loom/agent3
turns: 8, ended: complete
said: ...
settled: everything the merge left
base: 212ab3787fa38688dd3d1163f54deab8103b87f6
patch: /home/me/out/agents/20260903-215341-2-31577/target-repo/sandbox/patch-20260903-215341-2-31577.diff (0.6 KB, 2 file(s))
applied to the working tree: 2 file(s), over 212ab3787fa3. Commit them yourself
```

`base` is the commit that copies the working tree. When the merge settles, the union is applied to the working tree and the union branch is removed. When it is unsettled, nothing is applied, the exit status is not 0, and the union stays on `codetwine-loom/sandbox/<run-id>` ([What `integrate` leaves](limitations.md#what-integrate-leaves)). The union is a merge commit whose parents are the base and each branch. It carries the same trailers: `Verdict` is `settled` or `unsettled`, `Why` is what the checks still list, and the merge stage's decisions are kept for the next round.

Conflict markers are in git's `zdiff3` form: they hold the original lines as well as both sides, and name the branch each side comes from. Merge drivers from your git configuration are not started. Broken references are found by comparing codetwine's analysis of the base, of each branch and of the union, so only references the analysis resolves are covered ([limitations](limitations.md#only-references-the-analysis-could-resolve-to-the-referenced-file-show-up-as-broken)). The test check counts only failures the merge target does not have ([limitations](limitations.md#the-test-gate-of-integration-compares-with-the-merge-target)).

The merge stage reads each branch's request (the commit body) and what that agent said it did (`Codetwine-Loom-Did`) from the commits; it does not see other agents' output directories. On every turn the tool shows it what the checks still list. Besides the reading and rewriting tools, it has `read_branch_file(branch, path)`, `run_test` and `record_decision(what, why)`; only the reasons recorded with `record_decision` reach the union commit and a later `--resume`. While checks remain, each `COMPLETE()` is sent back, up to `INTEGRATE_MAX_REFUSAL` times (a `PRESET` column; 2 for medium).

`--resume <union-branch>` carries on from a union an earlier merge left unsettled. It starts from the union's tree, so what the earlier merge stage fixed remains, merges the union's parents again and runs the checks again. Add `--branch` to combine more branches. For a union whose merge settled but whose later stages did not, it does not merge again and runs from the unsettled stage. `--request` says what the branches were for; without it, the request the commits hold is used when they all hold the same one. `--engine`, `--image`, `--exclude` and `--doc-create` work as for `sandbox` and `run` ([Arguments](usage.md#arguments)). `--here` runs the merge directly in the current repository, without a container; it is what the container runs, and you do not normally pass it ([`integrate --here` writes to the working tree directly](limitations.md#integrate---here-writes-to-the-working-tree-directly)).

## Before a change goes into the working tree

Every result that `sandbox`, `run` or `integrate` applies goes through these checks. A result that touches any of the following is not applied at all; the run prints the paths, the location of the diff and the branch, and exits with code 1:

- an untracked `.env`, in any directory and any letter case (changes to a tracked `.env` are applied);
- a path git ignores in the working tree (inside `.venv/`, `node_modules/` and the like), including new files hidden by rules the change adds to `.gitignore`, and tracked files marked `skip-worktree` or `assume-unchanged`;
- a path inside a nested repository (a submodule or a directory holding `.git`), or a submodule's registration or commit;
- a path whose `filter` attribute points to a filter driver with a command in git's configuration (git-lfs and the like);
- a place git reads as configuration or runs as a command: git's configuration files and what they include, the files `core.attributesFile` and `core.excludesFile` name, the hooks directories (`.git/hooks`, `core.hooksPath` such as `.husky/`), and files and directories that a git setting or a variable such as `EDITOR` or `PATH` names as a command. Names are matched loosely: with `filter.x.clean = python3 scripts/clean.py`, everything below `scripts/` is protected, and `core.editor = code --wait` protects `code/` and `code.py`. When the top of the working tree is in `PATH`, nothing is applied. A file a hook merely calls is not covered ([A run that ends with exit status 0 writes to the working tree](limitations.md#a-run-that-ends-with-exit-status-0-writes-to-the-working-tree)).

Applying is all or nothing. The current contents of the touched paths are saved first. When applying fails partway or the command is stopped (Ctrl-C, `SIGTERM`, `SIGHUP`), the saved state is restored and the run counts as not applied. When the process is killed while applying, the save remains, and the next command that applies a change to that repository prints `a command ended before it was through putting a change into the working tree of …` with the save's location. When you edit the working tree during a run, the change may not apply; then nothing is written, the diff and the branch are printed, and the exit code is 1. A change not applied for any of these reasons does not make `run` redo the request.

A copy of what was applied is kept at `<OUTPUT_DIR>/agents/.repo/<repo-name>/land-last.diff`, and the last line prints the command that takes it back out (`git -C <target-repo> apply --binary --whitespace=nowarn --no-ignore-whitespace -R …`). This applying is the only time the subcommands that start containers write to your working tree.

For a merge (`run`, `integrate`), only a settled result is applied. When the machine's checks do not pass, nothing is applied, even when the merge stage explained why something cannot be cleared; the reason stays in the commit as `Codetwine-Loom-Why`. When the stages after the merge (`TEST_CREATE`, `REFACTOR`, `TEST_REFACTOR`) ran, the result is applied up to the last settled stage. When the test-writing stage is unsettled, only the union is applied. To apply a result that was not applied, read it and use the commands in [What a run that did not pass leaves](#what-a-run-that-did-not-pass-leaves) with the union branch.

When `--resume` carries on from a later stage of a settled union, it first checks whether the changes the earlier run applied are still in the working tree. When they all are, it applies only what the stage added; when none are, it applies everything from the base; when only some are, it writes nothing and prints the `git diff` command that shows what the stage settled.
