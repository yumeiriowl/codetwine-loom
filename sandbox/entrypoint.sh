#!/bin/sh
# The container side of the sandbox subcommand.
#
# The target repository is mounted read-only at CTLOOM_SRC. This clones it to CTLOOM_WORK,
# checks the clone out at CTLOOM_BASE, leaves what CTLOOM_BUILD_OUTPUT matches out of the
# clone's git, rebuilds the index over that state when CTLOOM_REBUILD_INDEX is 1, runs
# the tool over the clone with whatever arguments follow, runs the formatting command
# over the files that changed when CTLOOM_FORMAT is 1, and writes what changed to
# CTLOOM_PATCH as a git patch. Nothing outside CTLOOM_WORK, CTLOOM_OUT and the temporary
# directory is written.
#
# The tool is run from CTLOOM_RUNDIR, not from the clone.
#
# The patch is written against CTLOOM_BASE, so a commit made in the clone is in it too;
# integrate, which decides the commit its union is written against, has it written
# against HEAD. The patch and the summary beside it are written whether the run
# succeeded or not and whatever git could not add, and the run's exit status is this
# script's exit status. When the patch or the summary cannot be written, neither is
# left and the exit status is the run's, or 1 where that is 0.
#
# CTLOOM_PATCH.meta.json:
#   {"base": CTLOOM_BASE, "exit_status": the tool's, "files": how many the patch touches,
#    "unadded": [the paths git could not add, as git shows them],
#    "nested": [the repositories of their own the run left in the clone, whose files
#               are not in the patch],
#    "unseen": [the directories the index holds as submodules that hold a file the
#               patch does not: with no .git, or with a .git whose status is not
#               clean],
#    "moved_head": the commit HEAD ended at, only when the run moved it}
set -eu

: "${CTLOOM_SRC:=/src}"
: "${CTLOOM_WORK:=/work/repo}"
: "${CTLOOM_PATCH:=/out/patch.diff}"
: "${CTLOOM_OUT:=/out}"
: "${CTLOOM_REBUILD_INDEX:=0}"
: "${CTLOOM_FORMAT:=0}"
: "${CTLOOM_RUNDIR:=/tmp}"

if [ -z "${CTLOOM_BASE:-}" ]; then
    echo "ctloom-sandbox: CTLOOM_BASE is unset. Pass the commit the clone is to start at" >&2
    exit 2
fi

# git reads no configuration, no ignore file and no attributes file from HOME or
# XDG_CONFIG_HOME, runs no hook and copies no template into a repository it makes. The
# settings reach every git started below
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null
export GIT_CONFIG_KEY_1=init.templateDir GIT_CONFIG_VALUE_1=
export GIT_CONFIG_KEY_2=core.excludesFile GIT_CONFIG_VALUE_2=/dev/null
export GIT_CONFIG_KEY_3=core.attributesFile GIT_CONFIG_VALUE_3=/dev/null

# == Step 1: copy the repository to somewhere writable ===
git clone --no-hardlinks --quiet "$CTLOOM_SRC" "$CTLOOM_WORK"

# A commit no branch reaches is not carried over by the clone. Say so rather than
# starting the run at whatever the clone's HEAD happens to be
if ! git -C "$CTLOOM_WORK" cat-file -e "$CTLOOM_BASE^{commit}" 2>/dev/null; then
    echo "ctloom-sandbox: $CTLOOM_BASE is not in the clone of $CTLOOM_SRC. Only a commit some" \
         "branch reaches is copied by git clone" >&2
    exit 2
fi
git -C "$CTLOOM_WORK" checkout --quiet --detach "$CTLOOM_BASE"

# Not run from inside the clone. Every command below is given the clone's path, so the
# working directory is one whose contents are thrown away: a relative path written from
# the model's REPL lands there instead of in the repository, which is what the
# instructions state
cd "$CTLOOM_RUNDIR"

# The primary language's build_output_tuple entries that do not start with "/"
# (CTLOOM_BUILD_OUTPUT, one .gitignore pattern per line) are excluded in the clone's own
# git: what they match is not listed as changed, not analysed and not in the patch. The
# repository's own ignore files are not touched
if [ -n "${CTLOOM_BUILD_OUTPUT:-}" ]; then
    exclude_file=$(git -C "$CTLOOM_WORK" rev-parse --git-path info/exclude)
    case "$exclude_file" in /*) ;; *) exclude_file="$CTLOOM_WORK/$exclude_file" ;; esac
    mkdir -p "$(dirname "$exclude_file")"
    printf '%s\n' "$CTLOOM_BUILD_OUTPUT" >>"$exclude_file"
fi

# == Step 2: make the index describe that state ==========
# Only the definitions and the line numbers are rebuilt. No LLM is called, and the
# design documents already in CTLOOM_OUT stay as they are
if [ "$CTLOOM_REBUILD_INDEX" = "1" ]; then
    python -P -m codetwine_loom build --project "$CTLOOM_WORK" --output-dir "$CTLOOM_OUT"
fi

# == Step 3: run the tool over the clone =================
set +e
python -P -m codetwine_loom "$@"
status=$?

# == Step 3b: run the formatting command over what changed =
# Its own exit status is logged and does not change the run's
if [ "$CTLOOM_FORMAT" = "1" ]; then
    python -P -m codetwine_loom format --project "$CTLOOM_WORK" --output-dir "$CTLOOM_OUT"
fi
set -e

# == Step 4: write out what changed ======================
# Nothing here ends the script: the patch and its summary are written either way
set +e
mkdir -p "$(dirname "$CTLOOM_PATCH")"

# Print the lines of $1 as a JSON array of strings
json_list() {
    if [ -z "$1" ]; then
        printf '[]'
        return
    fi
    printf '[%s]' "$(printf '%s\n' "$1" \
        | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$/"/' | paste -s -d, -)"
}

# The commit the patch is written against, and where HEAD is when the run moved it
patch_base=$CTLOOM_BASE
moved_head=
if [ "${1:-}" = integrate ]; then
    patch_base=HEAD
else
    head=$(git -C "$CTLOOM_WORK" rev-parse HEAD 2>/dev/null)
    if [ -n "$head" ] \
            && [ "$head" != "$(git -C "$CTLOOM_WORK" rev-parse "$CTLOOM_BASE^{commit}")" ]; then
        moved_head=$head
        echo "ctloom-sandbox: HEAD is at $head and the run started at $CTLOOM_BASE; the patch" \
             "is written against $CTLOOM_BASE" >&2
    fi
fi

# A path git cannot add does not keep the rest out of the patch. Where git stops at
# one path and adds none, the changed paths are added one at a time
git -C "$CTLOOM_WORK" -c core.safecrlf=false add -A --ignore-errors
if [ $? -eq 128 ]; then
    git -C "$CTLOOM_WORK" ls-files -z --modified --deleted --others --exclude-standard \
        | GIT_LITERAL_PATHSPECS=1 xargs -0 -r -n 1 \
            git -C "$CTLOOM_WORK" -c core.safecrlf=false add -A --ignore-errors --
fi
unadded=$( { git -C "$CTLOOM_WORK" ls-files --others --exclude-standard
             git -C "$CTLOOM_WORK" diff --name-only --ignore-submodules=all; } | sort -u)
if [ -n "$unadded" ]; then
    echo "ctloom-sandbox: git could not add these paths, so they are not in the patch:" >&2
    printf '%s\n' "$unadded" >&2
fi

# A repository of its own inside the clone is added as its commit alone
tab=$(printf '\t')
nested=$(git -C "$CTLOOM_WORK" diff --cached --raw --no-renames \
        --ignore-submodules=untracked --submodule=short "$patch_base" \
    | sed -n "s/^:[0-7]* 160000 [0-9a-f.]* [0-9a-f.]* A$tab//p")
if [ -n "$nested" ]; then
    echo "ctloom-sandbox: these directories are repositories of their own, so the files" \
         "in them are not in the patch:" >&2
    printf '%s\n' "$nested" >&2
fi

# A directory the index holds as a submodule is shown as its commit alone, so a file
# there is in no patch when the directory holds it and no .git, or holds a .git whose
# own status is not clean or cannot be read
unseen=$(git -C "$CTLOOM_WORK" ls-files --stage -z \
    | sed -z -n "s/^160000 [0-9a-f]* [0-3]$tab//p" \
    | xargs -0 -r sh -c 'cd "$1" && shift && for path; do
            if [ -e "./$path/.git" ] || [ -L "./$path/.git" ]; then
                left=$(git --git-dir="./$path/.git" --work-tree="./$path" \
                    status --porcelain 2>/dev/null) || left=unread
            else
                left=$(find "./$path" ! -type d -print -quit 2>/dev/null)
            fi
            [ -z "$left" ] || git ls-files -- ":(literal)$path"
        done' sh "$CTLOOM_WORK")
if [ -n "$unseen" ]; then
    echo "ctloom-sandbox: git shows these submodule directories as their commit alone," \
         "so the files written in them are not in the patch:" >&2
    printf '%s\n' "$unseen" >&2
fi

# End without a patch and without a summary, with a non-zero exit status
no_patch_exit() {
    echo "ctloom-sandbox: $1, so no patch is left" >&2
    rm -f "$CTLOOM_PATCH" "$CTLOOM_PATCH.meta.json"
    [ "$status" -ne 0 ] || status=1
    exit "$status"
}

# A repository of its own is shown as its commit whatever .gitmodules sets it to ignore
git -C "$CTLOOM_WORK" diff --cached --binary --ignore-submodules=untracked --submodule=short \
        "$patch_base" >"$CTLOOM_PATCH" \
    || no_patch_exit "git could not write the patch"
files=$(git -C "$CTLOOM_WORK" diff --cached --name-only --ignore-submodules=untracked \
        --submodule=short "$patch_base" | wc -l | tr -d ' ')
printf '{"base": "%s", "exit_status": %d, "files": %s, "unadded": %s, "nested": %s, "unseen": %s%s}\n' \
    "$CTLOOM_BASE" "$status" "${files:-0}" "$(json_list "$unadded")" "$(json_list "$nested")" \
    "$(json_list "$unseen")" "${moved_head:+, \"moved_head\": \"$moved_head\"}" \
    >"$CTLOOM_PATCH.meta.json" \
    || no_patch_exit "the summary of the patch could not be written"

exit $status
