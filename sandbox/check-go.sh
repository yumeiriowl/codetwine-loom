#!/bin/sh
# The default static check for Go. Runs go vet over every package of the module in the
# current directory, test files included, and prints its findings, one
# "<path>:<line>[:<column>]: <message>" line each, with paths relative to the current
# directory. An import cycle is printed as a finding of go.mod (go.work where there is no
# go.mod) and two package clauses in one directory as a finding of the second file go
# names. Prints nothing when there is no go.mod and no go.work at the top.
#
# Exits 127 when go is not installed, when go's output holds a line saying the module is
# not set up for this go or a module cannot be fetched, or when go fails without a
# finding, with go's output on stderr.
set -u

if ! command -v go >/dev/null 2>&1; then
    echo "ctloom-check-go: go does not exist" >&2
    exit 127
fi
[ -f go.mod ] || [ -f go.work ] || exit 0
mod_file=go.mod
[ -f go.mod ] || mod_file=go.work

vet_out=$(go vet ./... 2>&1)
vet_status=$?
finding_out=$(printf '%s\n' "$vet_out" | awk -v root="$PWD" -v mod_file="$mod_file" '
function emit(text) {
    if (!(text in seen)) {
        seen[text] = 1
        print text
    }
}
function rel(path) {
    if (path == root) return "."
    if (index(path, root "/") == 1) return substr(path, length(root) + 2)
    return path
}
# The packages of the import chain from the first one the last one repeats, joined by " -> "
function cycle_text(    first, i, text) {
    first = chain_count
    for (i = 1; i < chain_count; i++) {
        if (chain[i] == chain[chain_count]) {
            first = i
            break
        }
    }
    text = chain[first]
    for (i = first + 1; i <= chain_count; i++) text = text " -> " chain[i]
    return text
}
# The line without the "vet: " go vet repeats a finding after, its indent, and a leading
# "./" or root
{
    line = $0
    sub(/^vet: /, "", line)
    sub(/^[ \t]+/, "", line)
    sub(/^\.\//, "", line)
    line = rel(line)
}
# An import chain: "package <path>", then one "imports <path>" line each, the last ending
# in "import cycle not allowed"
line ~ /^package [^ ]+$/ {
    chain_count = 1
    chain[1] = substr(line, 9)
    next
}
line ~ /^imports [^ ]+/ && chain_count > 0 {
    split(line, word, " ")
    sub(/:$/, "", word[2])
    chain[++chain_count] = word[2]
    if (match(line, /: import cycle not allowed( in test)?$/)) {
        emit(mod_file ":1: " substr(line, RSTART + 2) ": " cycle_text())
        chain_count = 0
    }
    next
}
{
    chain_count = 0
}
# Two package clauses in one directory, as a finding of the second file
match(line, /found packages [^ ]+ \([^)]+\) and [^ ]+ \([^)]+\) in /) {
    head = substr(line, RSTART, RLENGTH)
    dir = rel(substr(line, RSTART + RLENGTH))
    split(head, part, /[()]/)
    emit((dir == "." ? "" : dir "/") part[4] ":1: " head dir)
    next
}
# A finding of a .go, go.mod or go.work file
line ~ /^([^ #\t][^:]*\.go|([^ #\t][^:]*\/)?go\.(mod|work)):[0-9]+(:[0-9]+)?: / {
    emit(line)
}')
# go's lines saying the module is not set up for this go, or a module cannot be fetched
not_set_up_pattern='^pattern [^ ]+: directory prefix [^ ]+ does not contain '
not_set_up_pattern="$not_set_up_pattern"'|^go: go\.mod file not found |^go: go\.mod requires go >= '
not_set_up_pattern="$not_set_up_pattern"'|: module lookup disabled by GOPROXY=off$'
not_set_up_pattern="$not_set_up_pattern"'|: missing go\.sum entry for |^go: .*: dial tcp'
if printf '%s\n' "$vet_out" | grep -qE "$not_set_up_pattern"; then
    finding_out=""
fi
if [ "$vet_status" -ne 0 ] && [ -z "$finding_out" ]; then
    echo "ctloom-check-go: the module cannot be checked:" >&2
    printf '%s\n' "$vet_out" >&2
    exit 127
fi
[ -z "$finding_out" ] || printf '%s\n' "$finding_out"
exit 0
