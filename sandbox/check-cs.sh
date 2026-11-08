#!/bin/sh
# The default static check for C#. Builds the solution or project given as the argument
# and prints the compiler's error lines, one "<path>(<line>,<column>): error CS<code>:
# <message>" line each, each line once.
#
# Exits 127 when no solution or project is given, when dotnet is not installed, or when
# the build fails without reporting an error in a source file (a package it cannot
# restore, an SDK it does not have), with dotnet's output on stderr.
set -u

if [ "$#" -ne 1 ]; then
    echo "ctloom-check-cs: pass the one solution or project to build, e.g." \
         "CHECK_COMMAND=\"ctloom-check-cs src/App.sln\"" >&2
    exit 127
fi
if ! command -v dotnet >/dev/null 2>&1; then
    echo "ctloom-check-cs: dotnet does not exist" >&2
    exit 127
fi

build_out=$(MSBUILDTERMINALLOGGER=off dotnet build "$1" -nologo -v:q -clp:NoSummary 2>&1)
build_status=$?
error_out=$(printf '%s\n' "$build_out" | grep -E '^[^ ].*\([0-9]+,[0-9]+\): error ' \
    | sed -E 's/ \[[^]]*\]$//' | awk '!seen[$0]++')
if [ "$build_status" -ne 0 ] && [ -z "$error_out" ]; then
    echo "ctloom-check-cs: the project cannot be built:" >&2
    printf '%s\n' "$build_out" >&2
    exit 127
fi
[ -z "$error_out" ] || printf '%s\n' "$error_out"
exit 0
