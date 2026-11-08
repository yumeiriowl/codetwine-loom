#!/bin/sh
# The default launcher of run_project for C and C++. Copies the current directory, less
# .git, to a build directory emptied for each run (ctloom_make_setup in c-common.sh, which
# makes the makefile of an autotools project there) and runs make with the arguments
# there. The paths of the build directory in make's output are written as the repository's.
#
# Usage: ctloom-run-c <make argument>...
#
# Exits with make's status. Exits 127 when there is no makefile or configure at the top,
# when make is not installed, when the copy failed, or when the makefile cannot be made,
# with the reason on stderr.
set -u
. "$(dirname "$(readlink -f "$0")")/c-common.sh"

if [ -z "$(ctloom_make_source)" ]; then
    echo "ctloom-run-c: there is no makefile or configure in $(pwd); set RUN_COMMAND" \
         "to run a program built another way" >&2
    exit 127
fi
ctloom_make_setup ctloom-run-c 127
ctloom_make_in "$build_dir" "$(pwd)" --no-print-directory "$@"
printf '%s\n' "$make_out"
exit "$make_status"
