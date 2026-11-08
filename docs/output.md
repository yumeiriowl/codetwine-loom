# Output

This page lists the files this tool writes and the format of the ones you read. Step numbers refer to [Processing flow](processing-flow.md).

## Contents

- [Output files](#output-files)
  - [Files](#files)
- [Output schema](#output-schema)
  - [`code_index.sqlite`](#code_indexsqlite)
  - [Return value of `search_scope()`](#return-value-of-search_scope)
  - [`edit_history.json`](#edit_historyjson)
  - [`overview.json`](#overviewjson)
  - [Run record (`record/<run-id>.json`)](#run-record-recordrun-idjson)

## Output files

`build` writes to `<OUTPUT_DIR>/<repo-name>/`, where `<repo-name>` is the directory name of the target repository. `sandbox`, `run` and `integrate` do not work in it directly: each agent gets its own copy under `<OUTPUT_DIR>/agents/` ([Running several agents](container.md#running-several-agents)). `agents/.repo/<repo-name>/` holds what this tool keeps across runs, and `agents/.host/<agent>/<repo-name>/` holds what only the host writes; neither is passed into containers.

Subcommands that call the model directly on the host (`edit`, `answer`, `overview`, and `search` when it makes the overall overview) write their logs to `<repo-name>/turns/`. `edit` records each change in `<repo-name>/edit_history.json`, unless `EDIT_HISTORY=false` or `--dry-run`. codetwine also keeps a copy of each source file under `<repo-name>/`; the editing tools never change these copies.

```text
out/
  <repo-name>/                    made by build; agents are copied from here
    project_knowledge.sqlite
    analysed_commit.json
    code_index.sqlite
    overview.json                 only when the overall overview was made
    edit_history.json             records of edit on the host
    turns/                        logs of runs on the host
  agents/
    .repo/<repo-name>/            kept across runs; not copied to agents
      cache/                      package cache (/cache in the container)
      land-last.diff              the diff the last command applied to the working tree
      record/<run-id>.json        the record of one run of run, sandbox or integrate
    <agent>/<repo-name>/          one agent's records and patches (/out in the container)
      edit_history.json
      sandbox/patch-<run-id>.diff
      turns/
    .host/<agent>/<repo-name>/    host side; not passed into the container
      sandbox/land-<run-id>.diff
      sandbox/stdout-<run-id>.txt
      turns/                      the broker's logs
```

### Files

| File | Description |
|---------|------|
| `<repo-name>/project_knowledge.sqlite` | codetwine's analysis: definitions, call relations, design documents (step 1) |
| `<repo-name>/analysed_commit.json` | What the analysis represents: the tree state, the commit (empty when the analysed files differ from `HEAD`), the excluded names, the character encoding and the codetwine version (step 1) |
| `<repo-name>/code_index.sqlite` | The search index ([schema](#code_indexsqlite)) (step 2) |
| `<repo-name>/overview.json` | The overall overview ([schema](#overviewjson)). Also copied into each agent's copy |
| `<repo-name>/turns/<run-time>-<id>.log` | The host's log of `run`, `sandbox`, `integrate`, `warm` and the host subcommands |
| `agents/.repo/<repo-name>/cache/` | The package cache, kept across runs and read by all agents. `SANDBOX_CACHE_DIR` moves it. `cache.filled` next to it records what the cache was filled for |
| `agents/.repo/<repo-name>/land-last.diff` | The diff the last command applied to the working tree. `git -C <target-repo> apply --binary --whitespace=nowarn --no-ignore-whitespace -R` with it removes that change |
| `agents/.repo/<repo-name>/record/<run-id>.json` | The record of one run of `run`, `sandbox` or `integrate`, also written when the run stopped ([schema](#run-record-recordrun-idjson)). `history` lists them. The newest `DISPATCH_KEEP_RUN` records and those with branches to carry on from are kept |
| `agents/<agent>/…/edit_history.json` | The record of that agent's change ([schema](#edit_historyjson)) |
| `agents/<agent>/…/sandbox/patch-<run-id>.diff` | The change made inside the container, as a binary git diff from the base |
| `agents/<agent>/…/sandbox/*-<run-id>.json` | What the run inside the container handed back: the split of a request (`tasks-`), an answer (`answer-`, `{"answer": "..."}` or `{"error": "..."}`), or how a merge, test-writing or refactoring stage ended (`outcome-`: `verdict`, `why`, `did`, and the files or cases it changed or left) |
| `agents/<agent>/…/turns/<run-time>-<id>.jsonl` | One JSON line per turn: the time, the stage, the code the model wrote, its output, and the token usage |
| `agents/<agent>/…/turns/<run-time>-<id>.log` | That run's log, including the traceback of a run that ended with an exception |
| `agents/.host/<agent>/…/sandbox/land-<run-id>.diff` | The diff from the base to what was applied (or tried) to the working tree. When it did not apply, apply it by hand with `git -C <target-repo> apply --binary --whitespace=nowarn --no-ignore-whitespace` |
| `agents/.host/<agent>/…/sandbox/stdout-<run-id>.txt` | What the container printed, each line starting with the agent's label (when `AGENT_LOG_LEVEL` is not `DEBUG`) |
| `agents/.host/<agent>/…/turns/<run-time>-<id>.log` | The log of that agent's broker, such as failed model calls. Values the broker adds, such as `LLM_API_BASE`, are replaced with `<the broker's own>` |

## Output schema

### `code_index.sqlite`

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE defs (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,   -- "<file-path>::<definition-name>"
    name        TEXT NOT NULL,
    fold_name   TEXT,                   -- the name in upper case for a language that ignores case, NULL otherwise
    type        TEXT,
    file        TEXT NOT NULL,
    start_line  INTEGER,
    end_line    INTEGER,
    signature   TEXT,
    signature_line INTEGER,             -- the line number of signature
    is_constant INTEGER NOT NULL,       -- 1 for a definition left out of the search
    len         INTEGER NOT NULL        -- BM25 token count
);
CREATE INDEX idx_defs_name ON defs(name);
CREATE INDEX idx_defs_fold_name ON defs(fold_name);

CREATE TABLE files (
    id       INTEGER PRIMARY KEY,
    path     TEXT NOT NULL UNIQUE,
    summary  TEXT,
    doc_hash TEXT NOT NULL,             -- sha256 of the source when the design document was written. Empty when there is none
    len      INTEGER NOT NULL
);

CREATE TABLE calls (
    src  TEXT NOT NULL,                 -- the calling definition key
    dst  TEXT NOT NULL,                 -- the called definition key
    PRIMARY KEY (src, dst)
);
CREATE INDEX idx_calls_dst ON calls(dst);

CREATE TABLE postings (
    kind TEXT NOT NULL,                 -- "def" or "file"
    term TEXT NOT NULL,
    id   INTEGER NOT NULL,
    cnt  INTEGER NOT NULL,
    PRIMARY KEY (kind, term, id)
) WITHOUT ROWID;
```

| Field | Type | Description |
|-----------|-----|------|
| `meta.project` | string | The directory name of `PROJECT_DIR`. `load_index()` raises `ValueError` when it differs from its argument |
| `meta.commit` | string | The commit the analysis represents (empty when the analysed files differ from `HEAD`'s tree) |
| `meta.primary_ext` | string | The extension of the primary language: the language with the most files that have searchable definitions, preferring languages with a project file at the repository root. It decides the default test command |
| `defs.key` | string | `<file-path>::<definition-name>`. When a name appears several times in one file, `#<type>#<start-line>` is appended |
| `defs.signature` | string | The line that declares the definition. Shown in `scope["defs"]` and part of the search text |
| `defs.start_line` / `defs.end_line` | int | The range `read_definition` reads from the real file |
| `defs.is_constant` | int | 1 means left out of the search (`INCLUDE_CONSTANT` brings it back) |
| `files.summary` | string | The file's summary. Shown in `scope["file_summaries"]` |
| `files.doc_hash` | string | When it differs from the current file, the summary is not passed to the model |
| `calls` | table | Calls between definitions (`src` → `dst`), used by search step 3 |
| `postings` | table | The inverted index for each `kind` (`def` / `file`) |

The other `meta` keys record how and from what the index was built; when they no longer match, the index is rebuilt. `defs.file` and `files.path` use the analysis's notation (`<project-name>/<directory>/<stem>_<ext>/<basename>`); `search_scope()` converts them to repository-relative paths. The index holds neither the code of a definition nor its description.

### Return value of `search_scope()`

| Field | Type | Description |
|-----------|-----|------|
| `files` | string[] | The files to read, highest rank first (at most `SCOPE_FILE_K`) |
| `defs` | object[] | The definitions hit directly (at most `DEF_K`), each `{"key", "lines", "signature"}`. Test files' definitions come last |
| `expanded` | object[] | Definitions added by following call relations (at most `DEF_K`), each `{"key", "lines", "signature", "from"}`, where `from` is the key in `defs` it was reached from |
| `file_summaries` | object | `{file path: summary}`, each at most `SUMMARY_MAX_CHAR` characters; a cut summary ends with `... (N chars omitted)`. Files whose design document does not match the current file are left out |
| `callers` | object | `{definition key: [keys of the definitions calling it]}`, at most `DEF_K` each |
| `confidence` | string | `high` or `low` |
| `spread` | int | The number of files the directly hit definitions are spread over (0 to `DEF_K`) |
| `note` | string | One line when nothing was hit or step 1 failed. Empty otherwise |
| `trace` | object | Search details for debugging: `request`, `query`, `expanded`, `empty`, `expand_failed`, `top_score`, `top_raw`, `top_hit_terms` |

`defs[].lines` is `[start line, end line]` and can be passed to `read_source_file(path, start, end)`. To read one definition, use `read_definition(name)`; a definition inside a class can be looked up as `B.run`.

### `edit_history.json`

```json
{
  "schema": 1,
  "project": "string",
  "attempts": [
    {
      "n": 0,
      "at": "string",
      "request": "string",
      "did": "string",
      "files": {"added": ["string"], "modified": ["string"], "deleted": ["string"]},
      "verdict": "string",
      "signals": {"completed": true, "edits_landed": true,
                  "tests_ok": null, "check_ok": null, "final_ok": null,
                  "error": "string"},
      "why": "string",
      "last_run": "string",
      "verified": null,
      "verified_note": "string",
      "final_check": ["string"],
      "untooled": ["string"],
      "tests_changed": ["string"]
    }
  ]
}
```

| Field | Type | Description |
|-----------|-----|------|
| `attempts[].n` | int | The number of this change (from 1) |
| `attempts[].at` | string | When the run ended (ISO 8601, to the second) |
| `attempts[].request` | string | The request, cut at `EDIT_HISTORY_REQUEST_CHAR` |
| `attempts[].did` | string | The answer of `COMPLETE()` (or the late answer), cut at `EDIT_HISTORY_DID_CHAR` |
| `attempts[].files` | object | The files added, modified and deleted, up to `EDIT_HISTORY_MAX_FILE` each |
| `attempts[].verdict` | string | `failed`, `unverified`, `tests_red` or `no_change`. Never `passed` |
| `attempts[].signals` | object | What the verdict was decided from: `completed` (ended with `COMPLETE()`), `edits_landed` (changes remain), `tests_ok`, `check_ok` (the last automatic check), `final_ok` (the final static check and reference check), `error` (what stopped the run, or `""`). `null` means no result, not a pass |
| `attempts[].why` | string | One line giving the reason for `verdict` |
| `attempts[].last_run` | string | The end of the output of the last `run_project()` call, or `""` |
| `attempts[].verified` | string / null | `closed` when you chose to start fresh instead of carrying on from this entry; `null` otherwise |
| `attempts[].verified_note` | string | A note on `verified` |
| `attempts[].final_check` | array | What the final static check and reference check found. Absent when nothing |
| `attempts[].untooled` | array | Files changed without the editing tools. Absent when none |
| `attempts[].tests_changed` | array | The test files the run changed. Absent when none |

`request`, `did`, `files`, `verdict`, `why`, `last_run` and `untooled` are shown to the model at the start of the next run; `verified` and `verified_note` are not.

### `overview.json`

```json
{
  "schema": 3,
  "commit": "string",
  "file_digest": "string",
  "question": "string",
  "answer": "string"
}
```

| Field | Type | Description |
|-----------|-----|------|
| `commit` | string | The commit checked out when it was made (shown only in the log) |
| `file_digest` | string | A hash of the files in both the index and the working tree. When it differs from now, the overview is stale |
| `question` | string | The question asked. When it differs from the current one, the overview is stale |
| `answer` | string | The answer (the requirements), cut at `DISPATCH_ANSWER_MAX_CHAR` |

### Run record (`record/<run-id>.json`)

`agents/.repo/<repo-name>/record/<run-id>.json` is the record of one run of `run`, `sandbox` or `integrate`:

```json
{
  "id": "string",
  "command": "string",
  "request": "string",
  "start": "string",
  "end": "string",
  "elapsed_sec": 0.0,
  "exit_status": 0,
  "error": "string",
  "error_command": "string",
  "resume": ["string"],
  "files": ["string"],
  "token": {
    "call": 0,
    "input_token": 0,
    "output_token": 0,
    "cache_read_token": 0,
    "cache_write_token": 0,
    "total": 0,
    "limit": 0,
    "parts": [
      {"part": "string", "call": 0, "input_token": 0, "output_token": 0,
       "cache_read_token": 0, "cache_write_token": 0, "total": 0}
    ]
  },
  "report": "string",
  "result": {}
}
```

| Field | Type | Description |
|-----------|-----|------|
| `id` | string | The record's name, starting with the start time (the same as the file name) |
| `command` | string | `run` / `sandbox` / `integrate` |
| `request` | string | The request |
| `start` / `end` | string | The start and end times (ISO 8601, to the second) |
| `elapsed_sec` | float | The seconds it took |
| `exit_status` | int | The exit code ([exit codes](usage.md#exit-codes-and-how-a-run-ends-complete)) |
| `error` | string | The error that stopped the run. Empty when it did not stop |
| `error_command` | string | The command the `error` text points to, or empty |
| `resume` | string[] | The branches to pass to `--resume` to carry on. Empty when `sandbox` or `integrate` succeeded |
| `files` | string[] | The files applied to the working tree (repository-relative), including those of earlier rounds when `run` reran itself |
| `token.call` | int | The number of model calls |
| `token.input_token` / `token.output_token` | int | Input and output tokens. `cache_read_token` and `cache_write_token` are part of the input |
| `token.total` | int | Input plus output; the value compared with `MAX_TOKEN_PER_RUN` |
| `token.limit` | int | `MAX_TOKEN_PER_RUN` at that time (`0` means no limit) |
| `token.parts[]` | object[] | The breakdown per container (`part` is `<subcommand> <run-id>`). Calls the host made directly, such as the report, are only in the totals |
| `report` | string | The report printed at the end (only for a change made by `run`) |
| `result` | object | The subcommand's result. For `run`: each task's branch, verdict and exit code (`branches[]`), the merge (`integration`), the answer to a question (`answer`) and the things to confirm (`unclear`). For `sandbox` and `integrate`: the branch, the commit, the verdict, and the result of applying it to the working tree (`land`; inside `integration` for `run`). When the change did not apply, `land` holds the reason (`land_why`) and the command to apply it by hand (`land_command`). `token_stop` is set when `MAX_TOKEN_PER_RUN` was reached |
