#!/bin/sh
# The default test command for C and C++. Configures and builds the CMake project in the
# current directory with Ninja, keeping going past failures, and runs its tests with
# ctest --output-on-failure. When the build fails, the build output (Ninja's "FAILED:"
# lines and the compiler's errors) is printed and no test is run, so an executable left
# by an earlier build is never run in place of one that no longer builds.
#
# A directory with no CMakeLists.txt and a makefile, or an autotools configure to make
# one, is copied, less .git, to a build directory emptied for each run (ctloom_make_setup
# in c-common.sh, which runs the configure there), and make -k runs the makefile's test
# target there (its check target when it has no test target), with VERBOSE=yes where the
# copy has a Makefile.am, under which automake prints the logs of the tests that failed.
# The paths of the build directory in make's output are written as the repository's. A
# failed make whose every failed step is a command that is not installed
# (ctloom_make_not_installed) prints those lines on stderr, leaves out make's own "***"
# lines and exits 1, as a CMake project that cannot be configured does. Where the
# package cache is filled (CTLOOM_FILL=1) a makefile is not built: it prints nothing and
# exits 0.
#
# Usage: ctloom-test-c [--offline] [--no-make] (ctloom_read_args in c-common.sh).
#
# The CMake build directory is one per repository path under TMPDIR, kept between runs so
# a later run rebuilds only what changed. The makefile's is one per repository path too.
#
# Exits with ctest's status, or 1 when the build fails, and with make's status for a
# makefile. Exits 1 when the tools are not installed, when there is no CMakeLists.txt,
# makefile or configure at the top, when the makefile has no test or check target, or
# when the project cannot be configured, with the reason on stderr.
set -u
. "$(dirname "$(readlink -f "$0")")/c-common.sh"

ctloom_read_args "$@"

if [ ! -f CMakeLists.txt ]; then
    if [ -z "$(ctloom_make_source)" ]; then
        echo "ctloom-test-c: there is no CMakeLists.txt, makefile or configure in $(pwd);" \
             "set TEST_COMMAND to run the tests of a project built another way" >&2
        exit 1
    fi
    # Nothing is built where the package cache is filled (sandbox/warm.sh sets
    # CTLOOM_FILL)
    [ "${CTLOOM_FILL:-}" != 1 ] || exit 0
    ctloom_make_setup ctloom-test-c 1
    target=$(cd "$build_dir" && ctloom_make_target test check)
    if [ -z "$target" ]; then
        echo "ctloom-test-c: $makefile has no test or check target; set TEST_COMMAND to" \
             "run the tests of this project" >&2
        exit 1
    fi
    verbose=
    [ -f "$build_dir/Makefile.am" ] && verbose=VERBOSE=yes
    ctloom_make_in "$build_dir" "$(pwd)" -k "$target" $verbose
    if [ "$make_status" != 0 ] && missing=$(ctloom_make_not_installed "$make_out"); then
        printf '%s\n' "$make_out" | grep -v '^make\(\[[0-9]*\]\)\{0,1\}: \*\*\* '
        echo "ctloom-test-c: the makefile runs a command that is not installed here;" \
             "install it or set TEST_COMMAND:" >&2
        printf '%s\n' "$missing" >&2
        exit 1
    fi
    printf '%s\n' "$make_out"
    exit "$make_status"
fi
ctloom_need_tool ctloom-test-c 1 cmake ninja ctest

build_dir=$(ctloom_build_dir ctloom-test-c-ninja)
if ! ctloom_configure "$build_dir" "$offline"; then
    echo "ctloom-test-c: the project cannot be configured:" >&2
    printf '%s\n' "$configure_out" >&2
    exit 1
fi
if ! build_out=$(cmake --build "$build_dir" -- -k 0 2>&1); then
    printf '%s\n' "$build_out"
    exit 1
fi
exec ctest --test-dir "$build_dir" --output-on-failure
