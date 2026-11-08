#!/bin/sh
# Puts the packages the package.json in the current directory names into node_modules, so
# the tests of a JavaScript or TypeScript repository can load them.
#
# Usage: ctloom-install-js
#
# Nothing is done when there is no package.json, when it names no dependencies,
# devDependencies or optionalDependencies and no workspaces and there is no
# pnpm-workspace.yaml, or when node_modules holds the mark of an install made from the
# same package files (node_modules/.ctloom-installed, the checksum of the lock files,
# pnpm-workspace.yaml, bunfig.toml and every package.json outside node_modules, those of
# the workspaces included). Otherwise the packages are installed by the package manager
# the packageManager field of package.json names, or without one by npm where there is a
# package-lock.json or an npm-shrinkwrap.json, else by pnpm where there is a
# pnpm-lock.yaml, else by yarn where there is a yarn.lock, else by bun where there is a
# bun.lock, a bun.lockb or a bunfig.toml, else by npm, and the mark is written:
#   pnpm: "pnpm install --frozen-lockfile", and when that fails (a package.json changed
#     without its lock file) "pnpm install --no-frozen-lockfile", both with the store
#     under HOME (--store-dir) and dependencies' build scripts left unrun without failing
#     (--config.strict-dep-builds=false)
#   yarn 1: "yarn install --frozen-lockfile", and when that fails
#     "yarn install --no-lockfile", or "yarn install" when the lock file is made again
#   a later yarn: "yarn install --immutable", and when that fails "yarn install", both
#     into node_modules (YARN_NODE_LINKER) with the packages kept in yarn's cache outside
#     the repository (YARN_ENABLE_GLOBAL_CACHE)
#   npm: "npm ci" when there is a package-lock.json or an npm-shrinkwrap.json, and
#     "npm install --no-save --no-package-lock" when there is neither or when "npm ci"
#     fails, or "npm install" when the lock file is made again
#   bun: "bun install --frozen-lockfile" when there is a bun.lock or a bun.lockb (bun.lock
#     first), and "bun install --no-save" when there is neither or when that fails, or
#     "bun install" when the lock file is made again
# pnpm and yarn are run through corepack, at the version packageManager names. The
# package.json files are put back as they were before the install. The lock file is put
# back too, and one that was not there is taken away, except that when the install
# holding to it fails, it was there, and a package.json or a lock file differs from the
# commit the run started at (CTLOOM_BASE, else HEAD) or is a file git neither tracks nor
# ignores, it is made again from the package.json files and kept as the second install
# leaves it. When that second install fails, the lock file is put back.
# With npm_config_offline=true each installs from its cache alone (--offline, or
# YARN_ENABLE_NETWORK=0 for a later yarn).
#
# Exits 0 when the packages are in place or there are none. Exits 1, with the package
# manager's output and a line starting "ctloom-install-js: " on stderr, when node, npm or
# the package manager is not installed, when package.json cannot be read, or when the
# install fails.
set -u

NAME=ctloom-install-js
MARK=node_modules/.ctloom-installed
LOCK_FILES="package-lock.json npm-shrinkwrap.json pnpm-lock.yaml yarn.lock bun.lock bun.lockb"

[ -f package.json ] || exit 0

for tool in node npm; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "$NAME: $tool does not exist" >&2
        exit 1
    fi
done

if ! read_out=$(node -e '
const pkg = JSON.parse(require("fs").readFileSync("package.json", "utf8"));
let count = 0;
for (const field of ["dependencies", "devDependencies", "optionalDependencies"]) {
  const value = pkg && typeof pkg === "object" ? pkg[field] : null;
  if (value && typeof value === "object") count += Object.keys(value).length;
}
let workspaces = pkg && typeof pkg === "object" ? pkg.workspaces : null;
if (workspaces && !Array.isArray(workspaces)) workspaces = workspaces.packages;
if (Array.isArray(workspaces)) count += workspaces.length;
const field = pkg && typeof pkg.packageManager === "string" ? pkg.packageManager : "";
console.log(count + " " + field.split("@")[0]);
' 2>&1); then
    printf '%s\n' "$read_out" >&2
    echo "$NAME: package.json in $(pwd) cannot be read" >&2
    exit 1
fi
count=${read_out%% *}
manager_field=${read_out#* }
[ -f pnpm-workspace.yaml ] && count=$((count + 1))
[ "$count" -gt 0 ] || exit 0

case $manager_field in
pnpm|yarn|npm|bun) manager=$manager_field ;;
*)
    if [ -f package-lock.json ] || [ -f npm-shrinkwrap.json ]; then
        manager=npm
    elif [ -f pnpm-lock.yaml ]; then
        manager=pnpm
    elif [ -f yarn.lock ]; then
        manager=yarn
    elif [ -f bun.lock ] || [ -f bun.lockb ] || [ -f bunfig.toml ]; then
        manager=bun
    else
        manager=npm
    fi
    ;;
esac
if ! command -v "$manager" >/dev/null 2>&1; then
    echo "$NAME: $manager does not exist" >&2
    exit 1
fi

# Print the paths of the package.json files outside node_modules.
package_file_list() {
    find . -name node_modules -prune -o -name package.json -type f -print
}

# Print the checksum of the lock files, pnpm-workspace.yaml, bunfig.toml and the
# package.json files.
mark_of() {
    (cat $LOCK_FILES pnpm-workspace.yaml bunfig.toml 2>/dev/null
        package_file_list | LC_ALL=C sort \
            | while IFS= read -r file; do printf '%s\n' "$file"; cat "$file"; done) | cksum
}

if [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$(mark_of)" ]; then
    exit 0
fi

# Say whether a package.json or a lock file differs from the commit the run started at
# (CTLOOM_BASE, else HEAD) or is a file git neither tracks nor ignores. Outside a git
# work tree, or when git cannot compare, it says no.
dep_changed() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1
    { package_file_list
      for lock in $LOCK_FILES; do
          [ -f "$lock" ] && echo "$lock"
      done; } | while IFS= read -r file; do
        if [ -n "$(git ls-files --others --exclude-standard -- "$file" 2>/dev/null)" ]; then
            echo changed
        elif git ls-files --error-unmatch -- "$file" >/dev/null 2>&1; then
            git diff --quiet "${CTLOOM_BASE:-HEAD}" -- "$file" 2>/dev/null
            [ $? -eq 1 ] && echo changed
        fi
    done | grep -q changed
}

relock=""
dep_changed && relock=1

offline=""
[ "${npm_config_offline:-}" = true ] && offline=1
export COREPACK_ENABLE_DOWNLOAD_PROMPT=0

# Run the first install, and when it fails, the second, with the lock file put back as it
# was after it, or taken away when there was none. When relock is set and the lock file
# is there, the second install is the one that makes the lock file again, and what it
# writes there is kept unless it fails. Sets out and status.
# Usage: install_or_unlock <lock file> <install> <install not holding to the lock file>
#            <install making the lock file again>
install_or_unlock() {
    lock=$1
    out=$(eval "$2" 2>&1) && { status=0; return; }
    second=$3
    keep_lock=""
    note="so it is tried without it"
    if [ -n "$relock" ] && [ -f "$lock" ]; then
        second=$4
        keep_lock=1
        note="and a package file changed, so the lock file is made again"
    fi
    out=$(printf '%s\n%s: the install holding to %s failed, %s' "$out" "$NAME" "$lock" "$note")
    lock_copy=""
    if [ -f "$lock" ]; then
        lock_copy=$(mktemp) && cp "$lock" "$lock_copy" || { status=1; return; }
    fi
    second_out=$(eval "$second" 2>&1) && status=0
    if [ -z "$lock_copy" ]; then
        rm -f "$lock"
    elif [ -n "$keep_lock" ] && [ "$status" -eq 0 ]; then
        rm -f "$lock_copy"
    else
        cp "$lock_copy" "$lock" && rm -f "$lock_copy"
    fi
    out=$(printf '%s\n%s' "$out" "$second_out")
}

# The package.json files as they are before the install, which are put back after it
keep_dir=$(mktemp -d) || exit 1
package_file_list | while IFS= read -r file; do
    mkdir -p "$keep_dir/${file%/*}" && cp "$file" "$keep_dir/$file"
done

out=""
status=1
case $manager in
pnpm)
    pnpm_option="--store-dir '$HOME/.local/share/pnpm/store'"
    pnpm_option="$pnpm_option --config.strict-dep-builds=false"
    install_or_unlock pnpm-lock.yaml \
        "pnpm install --frozen-lockfile $pnpm_option ${offline:+--offline}" \
        "pnpm install --no-frozen-lockfile $pnpm_option ${offline:+--offline}" \
        "pnpm install --no-frozen-lockfile $pnpm_option ${offline:+--offline}"
    ;;
yarn)
    if ! version=$(yarn --version 2>&1); then
        printf '%s\n' "$version" >&2
        echo "$NAME: yarn could not be started" >&2
        rm -rf "$keep_dir"
        exit 1
    fi
    case $version in
    1.*)
        install_or_unlock yarn.lock \
            "yarn install --frozen-lockfile --non-interactive ${offline:+--offline}" \
            "yarn install --no-lockfile --non-interactive ${offline:+--offline}" \
            "yarn install --non-interactive ${offline:+--offline}"
        ;;
    *)
        export YARN_NODE_LINKER=node-modules YARN_ENABLE_GLOBAL_CACHE=true
        [ -n "$offline" ] && export YARN_ENABLE_NETWORK=0
        install_or_unlock yarn.lock "yarn install --immutable" "yarn install" "yarn install"
        ;;
    esac
    ;;
npm)
    npm_lock=""
    [ -f package-lock.json ] && npm_lock=package-lock.json
    [ -f npm-shrinkwrap.json ] && npm_lock=npm-shrinkwrap.json
    if [ -n "$npm_lock" ]; then
        install_or_unlock "$npm_lock" "npm ci --no-audit --no-fund" \
            "npm install --no-save --no-package-lock --no-audit --no-fund" \
            "npm install --no-audit --no-fund"
    else
        out=$(npm install --no-save --no-package-lock --no-audit --no-fund 2>&1) && status=0
    fi
    ;;
bun)
    bun_lock=""
    [ -f bun.lockb ] && bun_lock=bun.lockb
    [ -f bun.lock ] && bun_lock=bun.lock
    if [ -n "$bun_lock" ]; then
        install_or_unlock "$bun_lock" "bun install --frozen-lockfile ${offline:+--offline}" \
            "bun install --no-save ${offline:+--offline}" "bun install ${offline:+--offline}"
    else
        out=$(bun install --no-save ${offline:+--offline} 2>&1) && status=0
    fi
    ;;
esac
(cd "$keep_dir" && find . -name package.json -type f) | while IFS= read -r file; do
    cmp -s "$keep_dir/$file" "$file" || cp "$keep_dir/$file" "$file"
done
rm -rf "$keep_dir"
if [ "$status" -ne 0 ]; then
    printf '%s\n' "$out" >&2
    echo "$NAME: the packages package.json names could not be installed" >&2
    exit 1
fi
mark_of > "$MARK"
exit 0
