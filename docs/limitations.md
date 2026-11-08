# Limitations and notes

The limits and cautions to know before you run this tool on a repository or read a result. The README's [Limitations and notes](../README.md#️-limitations-and-notes) lists the main ones.

## Contents

- [Safety and secrets](#safety-and-secrets)
- [The target repository and the analysis](#the-target-repository-and-the-analysis)
- [Models, providers and cost](#models-providers-and-cost)
- [Runs and the working tree](#runs-and-the-working-tree)
- [Tests, checks and the change tools](#tests-checks-and-the-change-tools)
- [Merging and the stages after it](#merging-and-the-stages-after-it)
- [By language](#by-language)

## Safety and secrets

### The REPL is not a sandbox

The code the model writes runs in this tool's Python process, with its permissions. Path checks apply only when the model calls a tool; an `open()` or `shutil.rmtree()` the model writes directly is not checked. The only boundary is the container of `sandbox`, `run` and `integrate` ([Running in containers](../README.md#-running-in-containers)).

A run works from a temporary directory (`CTLOOM_RUNDIR`, default `/tmp`, removed at the end), so a relative `open("app.py", "w")` lands there and is discarded. An absolute path reaches the repository. Such writes cannot be stopped, but files changed outside the tools are listed in `untooled` of `edit_history.json` and in the commit's `Codetwine-Loom-Untooled` trailer. The tools do not limit writes to the files the search found: anything under `PROJECT_DIR` can be written.

### Do not start it with `python -m codetwine_loom` from inside the target repository

`python -m` imports from the current directory first, so the target repository's modules named like standard-library modules, or a `codetwine_loom/` directory there, run in place of this tool's. Start it with the `codetwine-loom` command. When a start loaded modules from the working directory, it stops with one line naming them, but they have already run by then. `python -m codetwine_loom` from the top of this repository is fine.

### A `.env` the target repository tracks is not read

When it runs inside the target repository, this tool does not read a `.env` that git tracks. When such a `.env` sets this tool's settings, every subcommand stops before starting ([How the values are read](settings.md#how-the-values-are-read)). An untracked `.env` is read, whoever put it there. Keep this tool's `.env` outside the target repository, and do not run the tool inside a directory you received together with its `.git` (an extracted archive, for example). A change from a container that touches an untracked `.env` is not applied ([Before a change goes into the working tree](container.md#before-a-change-goes-into-the-working-tree)).

### Programs that are started get no keys

The test command and `run_project` run the target repository's code. They get this process's environment without the settings this tool reads, except `COBOL_DIALECT` and `COBOL_COPYBOOK_DIR`, which the COBOL check needs. `PATH`, `HOME` and the cache directories are passed unchanged. Git and the container engine get the environment without the keys (`LLM_API_KEY`, `LLM_API_BASE`, `WEB_SEARCH_API_KEY`, `WEB_FETCH_API_KEY` and the like) and without `LLM_BROKER`. No started program gets the provider key variables of the configured `*_MODEL` models (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and so on), or the variables that change which repository git works on (`GIT_DIR`, `GIT_WORK_TREE` and the like).

Only those names are removed. Other secrets in the environment (`AWS_SESSION_TOKEN`, keys of providers no `*_MODEL` uses) are passed as they are, and the broker does not remove them from failure messages.

### What `sandbox` holds down is the destination, not the contents

The container has no network and no API keys, and its only way out is the broker's socket, so it cannot send data to a destination of its choice. But the container builds each model call, and the call can include anything the container has read. The repository's contents can therefore reach the configured provider.

### A `tcp://` broker has no authentication

A `unix://` socket is placed in a directory only you can open. A broker listening on `tcp://` lets anyone who can connect use the host's keys. When the containers run on the same machine, use `unix://`.

### `warm` and refills use the network

To fetch dependencies, a container with a network runs the test command once ([Fetching the dependencies ahead](container.md#fetching-the-dependencies-ahead-warm)). It holds no API keys and calls no model, but it connects to the registries the repository's build settings name, and a user and password written in the proxy URL are in its environment. `scripts.test` of `package.json`, `build.gradle`, `pom.xml` and `setup.py` are code the build tools run. In a refill during a change, those files come from the change, so the model chooses which packages are fetched and may have written that code. Only the number of refills is limited (`WARM_MAX_REFILL`); set it to `0` when you run requests you do not trust.

A fill runs on the copy of the working tree made when the command started. Changes an earlier command applied to the working tree, committed or not, therefore run in the next command's fill (`--resume` included). Check applied changes before the next command.

### The package cache is writable by every container, and containers with a network read it too

The package cache (`/cache`, also `HOME` in the container) is mounted read-write in the change containers (no network) and in the `warm` and refill containers (with a network), and it is kept across runs. Code in a change container can rewrite what tools read from `HOME` (fetched packages, `~/.yarnrc.yml`, Python's user site-packages, `~/.m2/settings.xml`, `~/.cargo/config.toml` and so on). A container with a network started later then reads it, and may run it. To keep caches apart per run, set `SANDBOX_CACHE_DIR`.

## The target repository and the analysis

### A repository whose directory name is `agents` cannot be handled

The output directory keeps the agents' files in `agents/`, so the analysis of a repository with that directory name cannot be stored. Every subcommand except `format`, `setup` and `broker` without `--project` stops before starting. Rename the repository's directory, or clone it into a directory with another name.

### The base is the working tree itself

Each run starts from a commit that copies the working tree, uncommitted changes and untracked files included ([The boundary `sandbox` sets up](container.md#the-boundary-sandbox-sets-up)). Files `.gitignore` ignores are not in it: ignored tests cannot be read, fixed or run, and `run_test()` returns `"did not run"`. Submodule contents are not in it either; their directories are empty in the container, so tests that need code inside them fail only there.

The container sees only that copy. Instruction files (`EDIT_GUIDE`, `TEST_GUIDE`, `REFACTOR_GUIDE`, `TEST_REFACTOR_GUIDE`) and COBOL copybook directories given as absolute paths, or as relative paths that leave the repository with `..`, cannot be read there; put them inside the target repository. The host warns at the start when an instruction-file setting points outside.

The build output listed as "not counted as changes" in [Defaults by language](settings.md#defaults-by-language) is left out only inside the container; on the host (`edit`, `--here`) only the repository's `.gitignore` decides. Build output in other places (a directory a build script names, say) goes into the patch unless `.gitignore` excludes it.

### What the analysis and the search cover

Files whose names are not UTF-8 are left out of the analysis and the index, with a warning naming them. `--exclude` and `BUILD_EXCLUDE` only take paths out; there is no setting to analyse only one directory. The index is rebuilt in full each time.

The index splits text into words at spaces and symbols, so make the design documents in English (leave codetwine's `OUTPUT_LANGUAGE` at its default, `English`). Without design documents, a definition is found only by the words of its path and its first lines, and the model is told to read only the signatures. Each language's reserved words and basic type names (Java's `public`, TypeScript's `number`, COBOL's `end` and so on) are not indexed from the code. Which kinds of definitions are searched is set by `INCLUDE_CONSTANT` ([Defaults by language](settings.md#defaults-by-language)).

The search marks its result `confidence: low` when the top definition's score is below `CONFIDENCE_MIN_SCORE` (default `0.3`) or fewer than 3 search terms hit it ([3. Search](processing-flow.md#3-search)); the model is then told to use `explore`. How well the threshold fits depends on the repository. Run `search` on a few requests and look at the `confidence:` line. When wrong candidates come with `high`, raise `CONFIDENCE_MIN_SCORE`.

### The cost of design documents grows with the repository

codetwine has an LLM write a design document for each file, so the calls and the cost grow with the number of files. Documents for every file are written only the first time; after that, only changed files and the files that depend on them are redone. Building without design documents calls no LLM. To keep the cost down, narrow the range with `--exclude`, or build the index without `--doc-create` first, see how the search does, and add design documents later with `build --doc-create`.

### Automatic rebuilds make no design documents

When the analysis no longer matches the code, it is remade without design documents and without calling an LLM. This happens inside the container when the base has moved, and on the host at the start of `search`, `answer`, `run-tests` and `edit`. Definitions, line numbers and call relations are updated; existing design documents are kept as they are. Summaries of rewritten files therefore stay old, and new files and definitions get none. To update the design documents, run `build --doc-create`, or pass `--doc-create` to the next `sandbox` or `run`.

### Only `run` asks when there is no analysis

A command other than `build` that finds no analysis makes one without design documents and carries on, without asking. Only `run` without `--doc-create`, at a terminal, asks once whether to make design documents too (Enter means no). For design documents, pass `--doc-create` or run `build --doc-create` first. No setting changes whether it asks.

## Models, providers and cost

### Cost limits

The broker counts model calls, `web_search`, `web_fetch` and cache refills, so code the model writes cannot exceed `LLM_MAX_CALL`, `WEB_SEARCH_MAX_CALL`, `WEB_FETCH_MAX_CALL` or `WARM_MAX_REFILL`. The broker also enforces each call's model, `max_tokens` and thinking settings ([The broker](container.md#the-broker-model-calls)). `RLM_MAX_TURN`, `EXPLORE_MAX_CALL`, `SEARCH_AGAIN_MAX_CALL` and `MAX_PATCH_ROUND` are kept inside the container, and the model's code can change or bypass them. Put your cost limit in `LLM_MAX_CALL` or `MAX_TOKEN_PER_RUN` (both `0`, no limit, by default), or at the provider. The call limits apply to each agent, so N agents of `run` can use N times as much.

`MAX_TOKEN_PER_RUN` caps the tokens of one run, all its agents together. Once it is reached, no new container starts and the broker refuses further calls; calls already in progress can push the total over. It does not stop the calls the host makes itself (the closing report), and it counts tokens, not money; design-document generation is not counted. When the budget runs out or the provider fails, the run keeps what it has written, records the verdict `failed` and makes no further calls.

### Write an endpoint that does not return redirects in `LLM_API_BASE`

Model calls do not follow redirects (3xx), with any provider. When the endpoint redirects (a gateway that forwards to another URL, `http` to `https`, and so on), every call fails. Write the URL that it redirects to.

### A local model server needs a context that holds the prompt and the reply

Keep each stage's output limit (`AGENT_MAX_TOKEN` and the `*_MAX_TOKEN` settings that follow it) within what the server's context (vLLM's `--max-model-len`) leaves for the reply. For a model litellm does not know, a call the server refuses for its length is sent again with half the output limit, down to 4096 tokens ([Models and model calls](settings.md#models-and-model-calls)). Ollama does not refuse: past its context (`num_ctx`) it silently drops the start of the prompt, where the instructions are. Set `OLLAMA_CONTEXT_LENGTH` on the server, or `PARAMETER num_ctx` in the Modelfile, before running.

### `web_search` needs a search provider and returns snippets only

`web_search` queries only the configured search provider: the model provider's own search with `WEB_SEARCH_PROVIDER=model`, or a search API. It never scrapes search engines, so there is no search without a provider. Without one, the model does not get the tool. Only the result snippets come back; to read a page, also set a fetch provider for `web_fetch` ([Reading a page of the search results](settings.md#reading-a-page-of-the-search-results-web_fetch)). Searches with `model` count toward `WEB_SEARCH_MAX_CALL`, not `LLM_MAX_CALL`.

## Runs and the working tree

### A run that ends with exit status 0 writes to the working tree

`sandbox`, `run` and `integrate` apply a result that ended with exit status 0 to the working tree, including a change whose verdict is `tests_red` or `unverified` ([verdicts](#whether-a-change-is-finished-cannot-be-decided-by-the-machine)). `run` and `integrate` also apply the union when the test-writing stage recorded cases the change does not meet; the run then ends with exit code 1. Nothing checks that what was applied is correct: read the diff and commit it yourself. The last line of the run prints a command that takes the applied change back out; it may fail when you have since edited the same lines. When the change cannot be applied, nothing is written ([Before a change goes into the working tree](container.md#before-a-change-goes-into-the-working-tree)).

The git this tool runs does not run hooks. The checks before applying refuse changes to hook directories and to programs that git filters and `core.fsmonitor` run, but not to files a hook calls (husky's `.husky/pre-commit`, pre-commit and lefthook settings, a script or `make` target a hook runs). Those run with the new contents when you commit, so in a repository with hooks, read the changes before `git commit`. Likewise, `git diff`, `git log -p` and `git merge` run the textconv, external diff, merge drivers and pager your git configuration names on the applied contents.

### Do not put runs side by side on the same working tree

When two `sandbox` runs overlap, the change of the one that ends later may not apply. Leave parallel work to `run`, and use `integrate` to combine branches people made. While `sandbox`, `run` or `integrate` is running in an output directory, a `build` that would remake the analysis and the `--doc-create` of `sandbox` or `run` stop in that output directory; wait, or use another `--output-dir`.

With `SANDBOX_NETWORK`, all agents of one `run` use the same service (the same database, say) at the same time. For tests that interfere with each other, set `DISPATCH_PARALLEL=1`. This tool does not start or clean up that service.

### Whether a change is finished cannot be decided by the machine

The `verdict` in `edit_history.json` is one of four, and never `passed`:

| Verdict | Meaning |
|---|---|
| `failed` | It crashed, no rewrite landed, it did not reach `COMPLETE()`, or the automatic check is red |
| `tests_red` | It reached `COMPLETE()` with the tests still red |
| `no_change` | It reached `COMPLETE()` without changing the repository |
| `unverified` | No red signal, but nothing shows the request was met |

A run may end while still reading the files it should change, or, in a change across several files, run out of turns with only some of them rewritten. Always have a person check the diff.

### Carrying on across runs

Only code applied to the working tree carries over to the next command; a container always starts fresh and does not know what an earlier run did. To continue an earlier run, use `run --resume <branch>`, which reads its work from the branch's commits ([Carrying on from an earlier run](usage.md#carrying-on-from-an-earlier-run-run---resume)). On the host, running `edit` again after a round that wrote something and ended `failed`, `tests_red` or `unverified` asks at a terminal whether to continue it (`--resume` / `--no-resume` skips the question). A change that did not pass in one run may pass when carried on.

To fix a defect found later in a finished change, give it as a new request and describe the defect itself. Only what the tool recorded (what it did, which files it touched, where it stopped) carries over.

## Tests, checks and the change tools

### `run_test()` is not a pass/fail gate

`COMPLETE()` is accepted even when the tests are red. It is sent back only for three reasons, none of them the test results: a planned file not yet touched, the program not starting in the automatic check (`SMOKE_ARG`), and new findings of the static check and the reference check (`--check`). The first two send it back once, the third up to `INTEGRATE_MAX_REFUSAL` times. When findings remain, the verdict is `failed` and `sandbox` leaves the change on the branch without applying it; carry on with `run --resume <branch>`, or turn the check off with `--no-check`. The analysis is remade at each check, so each `COMPLETE()` takes that time.

Green tests do not show that new code works: `run_test()` returns `"passed"` even when nothing was changed, and existing tests do not exercise new code. The model can run the program with `run_project`, but whether it does is up to the model. `SMOKE_ARG` makes the tool start the program after each edit ([Automatic check after each edit](settings.md#automatic-check-after-each-edit-smoke_arg)), which checks only that it runs. `FORMAT_COMMAND` runs after the last tests, so its rewrites are not tested.

### How test results are read

`run_test()` answers `"passed"`, `"failed"` or `"did not run"`. `"did not run"` means the tests could not be started (no target, a broken environment, no build tool, a timeout); it does not change the verdict. When no test command can run, the model does not get `run_test`.

A run that exits non-zero is red only when the output names failed tests or shows that it reached the code (a test summary, a compile error); otherwise it is `"did not run"`, and the round ends `unverified`. A run that exits 0 is red when its summary counts failed tests or a failed build (maven's `testFailureIgnore`, gradle's `ignoreFailures`), and `"did not run"` when its output says nothing ran (only skipped tests, `Ran 0 tests`, `0 passing`).

These rules read the default runners' normal output. A `TEST_COMMAND` whose tool prints none of these lines returns `"passed"` even with no tests, and adding `-q` to a maven or gradle command has the same effect. pytest at `-qq` or quieter prints no summary, so with `RUN_TEST=true` the subcommands that run tests refuse to start. JavaScript and TypeScript tests run with `CI=true`, so vitest and jest do not write missing snapshots, and a new `toMatchSnapshot()` test fails.

### The default static checks differ in strength by language

| Language | What the default check finds |
|---|---|
| Python | Undefined names only (ruff F821) |
| JavaScript | Undefined names only (ESLint `no-undef`). Misspellings that match a global of the browser, Node.js, Bun, jest or mocha (`name`, `status`, `run` and so on) are missed |
| TypeScript | Type and argument errors too (`tsc`), with the type definitions of Node.js and jest, and of Bun in a repository run by Bun (a `bun.lock`, `bun.lockb` or `bunfig.toml`, a `packageManager` of bun, or a `package.json` script that runs `bun` or `bunx`) |
| C / C++, Rust, C# | All compile errors (warnings are not counted) |
| COBOL | All errors of GnuCOBOL's syntax check, except inside `EXEC` and the names CICS, IMS, DB2 and MQ provide |
| R | Syntax errors and calls of undefined functions only |
| Java, Kotlin, SQL, BMS, JCL, CSD | No default check |

The defaults other than Python's are in the container image (the default image has all of them); on the host, or in an image without the language's toolchain, a warning is printed and nothing is checked. When the check tool fails, when C / C++ cannot be configured (libraries missing from the image, say), or when Rust's crates or C#'s packages are not in the package cache, a warning is printed and the check is skipped. C / C++ checks nothing, without a warning, when the repository has no `CMakeLists.txt`, top-level makefile or `configure`. Findings are compared by path and text, so an existing error whose text changes (a TypeScript error that names a type, for example) counts as new.

### What the rewriting tools refuse

The tools that write and delete files refuse paths outside `PROJECT_DIR`, under `OUTPUT_DIR`, under `.git`, paths git ignores and paths inside a nested repository; a path through a symbolic link is judged by its real path. Writing also refuses paths that are not regular files and files over 64 MiB. A file not in UTF-8 is written back in the encoding and BOM it was read with, and untouched lines keep their bytes and line endings. A file that cannot be decoded, or text its encoding cannot express, makes the write fail.

## Merging and the stages after it

### What `integrate` leaves

The branches given to `integrate` (`codetwine-loom/agent1` and the like) stay after the merge; remove them by hand when you no longer need them. When conflict markers or broken references remain, nothing is applied and the exit status is not 0, but the union is still committed to a branch for a person to check ([Combining several branches into one](container.md#combining-several-branches-into-one-integrate)). When three or more branches change the same lines, markers can nest; the work list says which branch each side comes from.

### `integrate --here` writes to the working tree directly

`--here` replaces the working tree's contents with the union (with `--onto`, `HEAD` also moves to the tip of that branch). It refuses, before reading any branch, when anything is uncommitted, when a changed tracked file is marked `assume-unchanged` or `skip-worktree`, or when an untracked item (an ignored `.env`, say) stands where a branch has a file. At the end it prints what was left and a `git -C <target-repo> checkout <original-branch> && git -C <target-repo> reset --hard` command to go back. Without `--here`, the merge happens in a container and the target repository is only read.

### The test gate of integration compares with the merge target

The merge target is `HEAD`, the commit that copies the working tree. Only tests that fail on the union and not on the merge target count, and that includes tests an agent broke on its own branch: the merge stage must fix them before `COMPLETE()` is accepted. Tests that do not run on the union count as left over. A merge target with no tests counts as passing. The gate is skipped when there is no test command, when the tests cannot be started on the merge target (a broken environment or a timeout, say), or when the merge target's run is red and one of the two runs gives neither the names nor the number of its failed tests. No union can settle when the merge target's run is red and stops before running every test (`-x`, `--maxfail`), or counts failed tests without naming each one ([7. Merging branches](processing-flow.md#7-merging-branches-integrate)).

### Only references the analysis could resolve to the referenced file show up as broken

A reference codetwine cannot link to its target is not reported as broken. In Python, for example, `from x import name` is seen, but `import x` followed by `x.name` is not. The merge stage's state shows how many references the analysis could not resolve. Definitions touched by two or more branches are found by matching diff lines to the definitions' line ranges, so calls within the same file are not seen. The test-writing stage finds the tests that refer to changed code the same way: tests that reach it through fixtures, HTTP or dynamic calls, and tests left out with `--exclude`, count as not referring to it.

### Tests can be rewritten, and written tests are not reviewed

The tools do not refuse to change or delete test files. The instructions of the change and merge stages say to fix only the assertions that pin down behaviour the request changes, not to weaken or remove tests to make them pass, and to remove the tests of code the request deletes. The model does not always follow them, so changed test files are listed in `tests_changed` of `edit_history.json`, in the commit's `Codetwine-Loom-Tests-Changed` trailer and in the log; check them.

The test-writing stage's gate checks that the cases are present, the references and the test results, not whether the expected values match the cases. The cases it decided are kept in `cases` of `outcome-<run-id>.json`. The tests it writes are applied without review; check the `written:` lines and the closing report before you commit.

### What the stages after the merge check

Only `run` and `integrate` run the test-writing and refactoring stages; `edit` and `sandbox` do not. The test-writing stage counts as tests only files under `TEST_PATH` (and, while `TEST_PATH` is unset, under the primary language's other test directories at the root, such as `test/` and `testing/` beside `tests/` for Python; see [Defaults by language](settings.md#defaults-by-language)) and files named like the primary language's test files. Tests elsewhere (a `test_x.py` next to the code, for example) are not counted. When code that disagrees with the request is outside a test task's files, either the task stays unsettled (the union is still applied, and the exit code does not change) or it records the gap in the commit and the run ends with exit code 1.

The refactoring stage sees behaviour only through the tests, so its gate passes when behaviour no test checks changes. Public names and signatures are protected only through references the analysis can trace. The test refactoring gate catches a whole test file disappearing, but not the loss of tests without case numbers when those tests also pass on the base. The full list of gate checks is in [8. Writing tests after the merge](processing-flow.md#8-writing-tests-after-the-merge-test-create) and [9. Refactoring after the merge](processing-flow.md#9-refactoring-after-the-merge-refactor-test-refactor).

## By language

### Tests of languages other than the main one run only where that language's dependency file is

When `TEST_COMMAND` is unset, other languages' default test commands run after the primary language's, when the command is in the image and the language's dependency file (`package.json`, `pom.xml`, `Cargo.toml` and so on) is in the repository. With dependency files in several directories, the tests run only in the one nearest the top that holds files of that language. The merge, test-writing and refactoring gates compare each command's results before and after the change. When `CHECK_COMMAND` is unset, the default static checks of every language in the index run the same way.

The test-writing stage writes tests in one language, the primary one. For a change that moves code to another language (it adds that language's first dependency file at the repository root, and touches definitions only in files the primary language's tests do not run), the test-writing and refactoring stages take the new language as primary, unless `TEST_COMMAND`, `TEST_PATH` or `TEST_TARGET_STYLE` is set. The refactoring stages then refactor only the files that language's tests run.

### Python's tests run on the image's Python

The default test command and `run_project` use the image's Python (3.12), or, where `.python-version` or `requires-python` names another version, one uv fetches for it while filling the package cache. When the pinned dependencies do not install or import on that Python, the tests are `"did not run"`. A directory with no `pyproject.toml`, or one without a `[project]` table, runs its tests in a separate environment with the repository's package and its requirement files. A `setup.py` package whose build imports something setuptools cannot load (Cython, for example) is not installed for the tests, with a warning; when its tests need the built package, set `TEST_COMMAND` to build it first ([Defaults by language](settings.md#defaults-by-language)).

### JavaScript and TypeScript tests run the whole `test` script

The default test command `npm test` runs everything the repository's `test` script runs: formatting, lint, type checks and builds as well as the tests. A step that needs the network fails in a container without one, and the tests after it do not run. For such a repository, set `TEST_COMMAND` to a script that runs only the tests (`npm run unit-test --silent`, say).

A JavaScript repository without `package.json` runs its test files with `node --test`, where tests written for another framework (jest's global `describe` and `expect`, for example) fail. A TypeScript repository without `package.json` runs no tests. In a repository using yarn 2 or later, a change to `package.json` without a matching `yarn.lock` cannot install in a container without a network, so its tests do not run.

### Java and Kotlin

`mvn -B -fae test` and `gradle test --continue` keep running modules that do not depend on a failed one; a run with skipped modules or a failed compile counts as one that stopped before running every test. Only Gradle's fetched dependencies are shared between containers; the rest of the Gradle home is rebuilt in each container, so the first build there is slower. The Gradle distribution `gradlew` downloads is not shared, so use the image's `gradle` in `TEST_COMMAND`.

### Rust

Only integration tests under `tests` count as test files (for the record of changed tests, and for where the test-writing stage puts new files); unit tests in `#[cfg(test)]` inside `src/` do not. `cargo test` runs every package of the workspace and keeps going when a target fails.

### C#

The default test command (`dotnet test`) and static check (`ctloom-check-cs`) use the one solution (`.sln`, `.slnx`) at the top, or else the one project (`.csproj`) there. With none, or more than one, the tests are `"did not run"` and the check is skipped; set `TEST_COMMAND` (`dotnet test src/App.sln`) and `CHECK_COMMAND` (`ctloom-check-cs src/App.sln`). When `global.json` asks for an SDK version the image lacks, neither runs ([Building the image](container.md#building-the-image)). A project builds only the `.cs` files under its directory, so new files must go under a project; tests written elsewhere do not run. Writing tests in a repository without a test project needs a new `.csproj` and a test framework's packages, which fails in a container without a network when they are not in the package cache. `run_project` works only when `RUN_COMMAND` is set (`dotnet run --project src/App`, say).

### C / C++ tests need CMake or a makefile with a test target

The default test command `ctloom-test-c` builds a repository with `CMakeLists.txt` and runs `ctest`, so only tests registered with `add_test` run (the test-writing stage can edit `CMakeLists.txt`). Without `CMakeLists.txt` but with a top-level makefile, or an autotools `configure` that makes one, it runs the makefile's `test` target (`check` when there is none) in a fresh copy of the repository, so each test run rebuilds everything and is slow on a large project. Makefiles are used only when C / C++ is the primary language. A command missing from the image makes the run `"did not run"`, but a missing header or library makes it red. For makefiles with neither target, other build systems, and repositories that need `./configure` options, set `TEST_COMMAND`.

For a makefile repository, `run_project` runs make in a copy of the repository (`ctloom-run-c`), so what it writes there is not kept; pass output paths with `{workspace}`. `ctloom-test-c` exists only in an image with the C toolchain (the default image has it), so on the host the tests are `"did not run"` unless `TEST_COMMAND` is set.

### R tests are testthat's `tests/testthat` only

The default test command `ctloom-test-r` runs the test files under `tests/testthat` with testthat. For other locations or frameworks (tinytest, RUnit), set `TEST_COMMAND`. Dependency packages are installed from source at their latest versions; the versions in `renv.lock` are not used. Packages that need system libraries the image lacks (`curl`, `xml2` and so on) cannot be installed, and the tests are `"did not run"`; build an image with the libraries added, or install the packages in the package cache by hand. `ctloom-test-r` and `ctloom-check-r` exist only in an image with the R toolchain (the default image has it), so on the host the tests are `"did not run"` unless `TEST_COMMAND` is set.

### COBOL, BMS, JCL and CSD

COBOL has no default test command or launcher. Unless `TEST_COMMAND` is set, the test-writing stage does not run (except for a change that moves code to another language), and `run_project` works only when `RUN_COMMAND` is set (`cobc -x -j -I copy src/MAIN.cbl`, say). A copybook is analysed as a file of its own; a program's uses of its items become references to the copybook's definitions. The stages show COBOL, BMS, JCL and CSD names to the model in upper case.

The rewriting tools do not refuse writes before column 7 or after column 72. The instructions tell the model to keep the fixed-format columns, and the static check reports what becomes a syntax error. The check covers only what GnuCOBOL can read: constructs only IBM Enterprise COBOL has, and lines GnuCOBOL cannot read in fixed format, count as existing findings, and CICS, DB2, IMS and MQ statements are not checked. In programs that include copybooks missing from the repository, undefined names are missed. Point to copybooks that `COPY` does not find with `COBOL_COPYBOOK_DIR` (a path inside the repository for commands that run in a container), and choose the dialect with `COBOL_DIALECT` ([Tests and the static check](settings.md#tests-and-the-static-check)).
