#!/bin/sh
# The default static check for C and C++. Builds the CMake project in the current
# directory with Ninja, keeping going past failures, and prints the compiler's error
# lines, one "<path>:<line>:<column>: error: <message>" line each. A directory with no
# CMakeLists.txt and a makefile, or an autotools configure to make one, is copied, less
# .git, to a build directory emptied for each run (ctloom_make_setup in c-common.sh), and
# make -k builds the makefile's default target there, with the paths of the build
# directory written as the repository's; a header that is not a file of the repository
# is left out, and a prerequisite make has no rule for is printed as a finding
# (ctloom_make_check_lines).
# Prints nothing when there is no CMakeLists.txt, makefile or configure at the top.
#
# Usage: ctloom-check-c [--offline] [--no-make] (ctloom_read_args in c-common.sh).
#
# The CMake build directory is one per repository path under TMPDIR, kept between runs so
# a later run rebuilds only what changed. The makefile's is one per repository path too.
# A call of an undeclared function is an error in the CMake build.
#
# Exits 127 when cmake, ninja or make is not installed, when the project cannot be
# configured, with cmake's output on stderr, or when the repository cannot be copied.
set -u
. "$(dirname "$(readlink -f "$0")")/c-common.sh"

ctloom_read_args "$@"

if [ ! -f CMakeLists.txt ]; then
    [ -n "$(ctloom_make_source)" ] || exit 0
    ctloom_make_setup ctloom-check-c 127
    ctloom_make_in "$build_dir" "$(pwd)" -k
    ctloom_make_check_lines "$makefile" "$make_out"
    exit 0
fi
ctloom_need_tool ctloom-check-c 127 cmake ninja

build_dir=$(ctloom_build_dir ctloom-check-c-ninja)
CFLAGS="${CFLAGS:-} -Werror=implicit-function-declaration"
export CFLAGS
if ! ctloom_configure "$build_dir" "$offline"; then
    echo "ctloom-check-c: the project cannot be configured:" >&2
    printf '%s\n' "$configure_out" >&2
    exit 127
fi
cmake --build "$build_dir" -- -k 0 2>&1 | grep -E ' (fatal )?error:'
exit 0
