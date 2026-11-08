#!/bin/sh
# The default static check for Rust. Checks every target of every package of the Cargo
# project in the current directory and prints the compiler's error lines, one
# "<path>:<line>:<column>: error[<code>]: <message>" line each. Prints nothing when
# there is no Cargo.toml at the top.
#
# Exits 127 when cargo is not installed, or when cargo fails without reporting an
# error in a source file (a manifest it cannot read, a crate it cannot fetch), with
# cargo's output on stderr.
set -u

if ! command -v cargo >/dev/null 2>&1; then
    echo "ctloom-check-rs: cargo does not exist" >&2
    exit 127
fi
[ -f Cargo.toml ] || exit 0

check_out=$(cargo check --workspace --all-targets --message-format short 2>&1)
check_status=$?
error_out=$(printf '%s\n' "$check_out" | grep -E '^[^ ].*:[0-9]+:[0-9]+: error')
if [ "$check_status" -ne 0 ] && [ -z "$error_out" ]; then
    echo "ctloom-check-rs: the project cannot be checked:" >&2
    printf '%s\n' "$check_out" >&2
    exit 127
fi
[ -z "$error_out" ] || printf '%s\n' "$error_out"
exit 0
