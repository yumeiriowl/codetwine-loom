# Changelog

## 0.1.0 - 2026-11-08

First public release.

### Added
- Subcommands to build the analysis and the search index, search, answer questions, change code on the host or in a container, and run whole requests (`run`)
- `run` splits a request into questions or tasks, runs agents in parallel, merges their branches, then writes tests and refactors the change
- Changes run in a container with no network and no API key, and are applied to the working tree without committing
- Definitions and call relations for the 15 languages codetwine analyses; every text file is searchable
- `search_scope()`, `create_agent()` and `run_edit()` for use as a Python library
- An Agent Skill (`skills/codetwine-loom/`) that lets a coding agent set up and use this tool
