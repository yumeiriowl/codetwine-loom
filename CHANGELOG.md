# Changelog

## 0.2.0 - 2026-11-17

### Added
- Go: definitions and call relations, `go test ./...` as the default test command with its output read for failing tests, packages that do not build, runs over no tests and modules that cannot be fetched, `go run` as the default launcher, and the static check `ctloom-check-go` (`go vet`)
- The image holds Go (`TOOLCHAIN=go`, `GO_VERSION`)

## 0.1.0 - 2026-11-08

First public release.

### Added
- Subcommands to build the analysis and the search index, search, answer questions, change code on the host or in a container, and run whole requests (`run`)
- `run` splits a request into questions or tasks, runs agents in parallel, merges their branches, then writes tests and refactors the change
- Changes run in a container with no network and no API key, and are applied to the working tree without committing
- Definitions and call relations for the 15 languages codetwine analyses; every text file is searchable
- `search_scope()`, `create_agent()` and `run_edit()` for use as a Python library
- An Agent Skill (`skills/codetwine-loom/`) that lets a coding agent set up and use this tool
