# Processing flow

This page describes each stage in order. The settings each stage reads are in [settings.md](settings.md).

## Contents

- [1. Preparing the analysis](#1-preparing-the-analysis)
- [2. Building the index](#2-building-the-index)
- [3. Search](#3-search)
- [4. Changing the code](#4-changing-the-code)
- [5. Dispatching a request (`run`)](#5-dispatching-a-request-run)
- [6. Answering questions (`answer`)](#6-answering-questions-answer)
- [7. Merging branches (`integrate`)](#7-merging-branches-integrate)
- [8. Writing tests after the merge (`test-create`)](#8-writing-tests-after-the-merge-test-create)
- [9. Refactoring after the merge (`refactor`, `test-refactor`)](#9-refactoring-after-the-merge-refactor-test-refactor)

## 1. Preparing the analysis

The analysis is codetwine's output, `<OUTPUT_DIR>/<project-name>/project_knowledge.sqlite`. `<project-name>` is the directory name of `--project`. The step stops when `--project` is not the top of a git working tree, when `OUTPUT_DIR` is inside `PROJECT_DIR` or contains it, and when the analysis in place belongs to another repository with the same directory name.

When there is no analysis, codetwine makes one. It covers the files git tracks and the untracked files `.gitignore` does not ignore. It leaves out symbolic links, paths that lead outside the repository, file names that are not UTF-8, and what `--exclude` removes. Warnings name the files that could not be read or whose character encoding was guessed. With `--doc-create`, the log shows `docs: N ok, M failed`, and the step stops when no design document at all could be made. `analysed_commit.json`, next to the analysis, records what the analysis represents.

When the analysis **does not represent the current working tree** (files or their contents changed, or the excluded names, `SOURCE_ENCODING` or the installed codetwine differ), every subcommand except `integrate` makes it again without asking. This calls no LLM and writes no design documents; the existing ones are kept. `build --doc-create` always runs codetwine, and writes design documents for files that have none, files that changed, and files that depend on them. `sandbox --doc-create` and `run --doc-create` do the same once at start-up.

When the output directory has no analysis, `sandbox`, `run` and `integrate` analyse the host's working tree and build the index before any container starts; the containers work from a copy of it. Only `run` asks, at a terminal and without `--doc-create`, whether to make design documents too (Enter means no). Inside a container, the analysis is made again (without design documents) only when it does not represent the commit the container starts from or was made by a different codetwine.

Note: in this step only codetwine calls an LLM, and only for design documents. Each file takes one call per section of the `DOC_TEMPLATE_PATH` template, plus one for the summary (more for long definitions). `run`'s question states the number of calls.

## 2. Building the index

The index, `code_index.sqlite`, is built from the analysis without calling an LLM. Every subcommand rebuilds it without asking when it is missing, was made from an older analysis, or was made by different rules (for example a changed `PATH_WEIGHT`).

The index holds one entry per definition and per file, and the call relations between definitions that codetwine found. Each definition gets a `signature`: the first line of its code that contains its name. The design documents are matched to definitions by their headings (such as `` `Store.close` ``), then by quoted names; a definition with no match gets the file's summary. Names in COBOL, BMS, JCL and CSD files match regardless of case.

Two BM25 indexes are written. The definition index holds the path words, the signature and the matched description. The file index holds the path words and the file summary. Identifiers are also split into their parts (`load_config` into `load` and `config`), and the language's reserved words are dropped from the code.

## 3. Search

**Step 1** converts the request into English search terms with one LLM call (`QUERY_EXPAND_MODEL`). An ASCII-only request skips this call, unless the overall overview is passed; then the overview is added to the prompt. File names, paths and identifiers written in the request are kept as search terms. When the call fails or returns no usable words, the request is searched as it is, and `note` says so.

**Step 2** ranks definitions with BM25, moves definitions in test files (`TEST_PATH`) to the back, and keeps the top `DEF_K`. **Step 3** follows the call relations one hop from each hit. A neighbour's score is the hit's score times `EXPAND_DECAY`, divided by how many definitions call the neighbour. The added definitions appear in `scope["expanded"]`. Definitions in test files are neither added nor expanded from.

**Step 4** groups the definitions by file and merges this ranking with the file index's ranking (reciprocal rank fusion). Files named in the request go to the head. When several files share that name, only those whose directory is also written are kept. Line ranges, signatures, file summaries and `spread` are attached, and callers are listed. `confidence` is `high` when the top definition's score reaches `CONFIDENCE_MIN_SCORE` and at least 3 search terms hit it ([what the search covers](limitations.md#what-the-analysis-and-the-search-cover)).

## 4. Changing the code

With `TWO_STAGE=true`, a read-only planning stage runs first and passes its plan to the change stage. The search result is attached to the request and is also available in the REPL as `scope`.

Each turn, the model writes one Python block. The block runs in the REPL, and the model sees its output. This repeats for up to `RLM_MAX_TURN` turns. The model reads with the reading tools, writes and deletes files with the change tools, runs the tests with `run_test()` and the program with `run_project()`. When it cannot narrow down the candidates, `explore("<aspect>")` runs a separate read-only loop that returns only the places to change. When `SMOKE_ARG` is set, the program is started after each turn that changed files, and the result is added to that turn's output. When the model finds that no change is needed, it ends with `COMPLETE("<why-not-needed>")`.

Note: every turn calls the LLM. Each `explore` call runs up to `EXPLORE_MAX_TURN` turns.

## 5. Dispatching a request (`run`)

Unless `--no-overview` is passed, `run` first prepares the analysis and the index on the host, then makes the overall overview (`overview.json`) in its own container when it is missing or stale. Every worker passes the overview to search step 1.

The dispatching stage searches as in section 3, then runs one read-only loop of up to `DISPATCH_MAX_TURN` turns. It does not change the repository and does not answer the request itself. It ends with one of `COMPLETE({"questions": [...]})`, `COMPLETE({"tasks": [...]})` or `COMPLETE({"unclear": [...]})`. When it runs out of turns, it is asked once more for the JSON alone.

For tasks, the host starts one `sandbox` run per task, up to `DISPATCH_PARALLEL` at once. Tasks that name the same definition run one after another, each on the commit the previous one left. Tasks that did not finish are handed out again. When all tasks have finished, their branches go to the merge (section 7). For questions, the host starts one `answer` run per sub-question and passes the answers to the merge of section 6. For things to confirm, the host starts no agents, prints them and exits with code 1. What the user sees is in [How `run` handles a request](usage.md#how-run-handles-a-request).

## 6. Answering questions (`answer`)

A worker searches as in section 3, then reads the code for up to `ANSWER_MAX_TURN` turns and ends with `COMPLETE("<answer>")`. It has the explorer's read-only tools, plus `web_search` when a search provider is set and `web_fetch` when a fetch provider is set too. It cannot run the program. The model is `ANSWER_MODEL`.

The merge searches with the original question, receives the workers' answers, and runs for up to `INTEGRATE_MAX_TURN` turns with the same tools (`INTEGRATE_MODEL`). It starts from the answers and reads the code only where they conflict, leave something open, or say it was not checked. When no worker answered, one run answers the original question instead. The answer is cut at `DISPATCH_ANSWER_MAX_CHAR` characters. A host `answer` reads the working tree; `run`'s workers read a clone of it.

## 7. Merging branches (`integrate`)

The base is the commit the given branches share. The branches are merged one at a time, and the result (the union) is written to the working tree of the container's clone. The machine then checks 6 things:

1. **The static check** (`CHECK_COMMAND`) reports no findings on the union that it did not report on the merge target (`HEAD`).
2. **No conflict markers are left**, in tracked or untracked files. Marker-like lines that the base or a branch already had are allowed.
3. **The merge broke no references**: everything that resolves on the base and on each branch alone still resolves on the union.
4. **The tests got no worse than on `HEAD`**. The tests also run on `HEAD`; tests that fail there too are ignored, and only new failures count. When `HEAD`'s run stopped early (`-x`, `--maxfail`), no union can settle, and the last line says to make the test command run every test or to fix the failing tests. This check is skipped when there is no test command or the tests cannot be started on `HEAD`; a `HEAD` with no tests counts as passing ([details](limitations.md#the-test-gate-of-integration-compares-with-the-merge-target)).
5. **Every definition a branch added or changed is in the union**, matched by name in any file. A name another branch removed or renamed is not required.
6. **Each conflicted file was read at least once** by the merge stage. A file missing from the working tree can be read from a branch with `read_branch_file`.

The machine also lists, for the model to read, definitions that two or more branches changed, and definitions one branch changed that another branch's change calls.

**When all 6 checks pass and the machine listed no definitions to read, the stage ends without calling the model.** A file holding a listed definition counts as unread until the model reads it. Otherwise only what is left over goes to the model as a work list. The same checks run at each `COMPLETE()`, which is sent back up to `INTEGRATE_MAX_REFUSAL` times while checks remain. The host commits the result as a merge commit whose parents are the merge target and each branch.

Note: the reference check runs the codetwine analysis once more on the union, without design documents (no LLM call).

## 8. Writing tests after the merge (`test-create`)

This stage runs in `run` and `integrate` after the merge settles, before anything is applied to the working tree. It runs in a container based on the union's commit and adds its own commit (`test-create: ...`) to the same branch. `--no-test-create` turns it off. Its settings and the lines it prints are in [Writing tests after the merge](settings.md#writing-tests-after-the-merge-test_create).

**Targets.** A first run lists the changed definitions (definitions of the union that the diff from the base touches), the tests that refer to them, and the tests that fail now. The failing tests become the baseline. When there are more changed definitions than `TEST_CREATE_MAX_DEFINITION`, the work is split into test tasks, each in its own container, run in parallel; target files that share a test file stay in one task. At the end, all tasks' tests go into one commit. Which files count as tests is set by `TEST_PATH` ([settings](settings.md#tests-and-the-static-check)).

**Two parts per task.** The first part decides the test cases. It sees the request, the task text, the first line of each changed definition and the existing tests, and can read only test files. It returns the cases as JSON, numbered `case_<number>`. The second part receives the cases and writes the test code, with the full reading and writing tools. Its instructions end with the repository's test instruction file (`TEST_GUIDE`). Each part runs up to `TEST_CREATE_MAX_TURN` turns.

**The gate.** At each `COMPLETE()` of the second part, the machine checks that:

- no file outside what the task may write has changed;
- every case is in a test file, or recorded with a reason;
- no test file it started with was removed;
- the static check and the reference check find nothing new;
- no test fails that did not fail before;
- each test file it wrote passes when run alone (where the build tool accepts paths), and the whole test suite runs it.

While a check fails, `COMPLETE()` is sent back, up to `TEST_CREATE_MAX_REFUSAL` times. When the code does not do what a case expects and the task cannot fix it within its files, the model records the case with `record_gap`; the report at the end lists these cases, and `run` and `integrate` exit with code 1.

**On the base.** At the end, the new tests also run on the base's tree, and the result is recorded as `red` (they fail there), `green` (they pass there too) or `unknown`.

Note: each gate check runs the codetwine analysis again without design documents (no LLM call).

## 9. Refactoring after the merge (`refactor`, `test-refactor`)

Two stages run after the test-writing stage, each in one container based on the head of the branch, each adding its own commit: **implementation refactoring** (`refactor`), then **test refactoring** (`test-refactor`). `--no-refactor` and `--no-test-refactor` turn them off. Their settings, what they look for, and the lines they print are in [Refactoring after the merge](settings.md#refactoring-after-the-merge-refactor--test_refactor).

**Targets.** The targets are what changed between the base and the head of the branch, so they include what the test-writing stage changed. Implementation refactoring may rewrite only the non-test files whose definitions changed; it adds no files. Test refactoring may rewrite the test files that were added or changed, and add test files. For a change that moves code to another language, only the files that language's tests run are targets ([details](limitations.md#tests-of-languages-other-than-the-main-one-run-only-where-that-languages-dependency-file-is)).

**The run.** Each stage first runs the tests; the failing tests become the baseline. When there are no targets, the tests cannot run, or the baseline cannot be read, the stage ends without calling the model. Otherwise the model works with the same tools as the second part of section 8, for up to `REFACTOR_MAX_TURN` turns. The writing tools refuse files that are not targets.

**The gate.** At each `COMPLETE()`, the machine checks that no other file changed, the static check and the reference check find nothing new, and no test fails that did not fail before. For test refactoring it also checks that no test file was removed, every case number is still present, each changed test file passes when run alone and is run by the whole suite, and the tests give the same result on the base's tree as before. While a check fails, `COMPLETE()` is sent back, up to `REFACTOR_MAX_REFUSAL` times. When a stage does not settle, the stages after it do not run.

Note: each gate check runs the codetwine analysis again without design documents (no LLM call).
