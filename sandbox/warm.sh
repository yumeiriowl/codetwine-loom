#!/bin/sh
# The container side of the warm subcommand.
#
# The target repository is mounted read-only at CTLOOM_SRC. This clones it to CTLOOM_WORK,
# checks the clone out at CTLOOM_BASE, copies in the dependency files of a change that is
# running (CTLOOM_LIVE) when there is one, and runs the tool over the clone with whatever
# arguments follow. Nothing is written back: the clone goes away with the scratch
# directory, and what is kept is whatever the toolchain left in the cache mount.
#
# git's configuration, ignore file and attributes file, the directory XDG_CONFIG_HOME
# names, and the user configuration of pip, npm and mvn's mavenrc are not read from the
# cache mount.
#
# Only the paths in CTLOOM_DEP_PATH (one per line, relative to CTLOOM_LIVE) are taken from
# CTLOOM_LIVE. The code that runs here is the clone's.
#
# In an image with a JVM toolchain, JAVA_TOOL_OPTIONS is given what ctloom-jvm-route makes
# of the proxy variables and SSL_CERT_FILE, and the dependencies Gradle fetched are kept
# in the directory the other containers read them from (ctloom-gradle-share).
#
# The run's exit status is this script's exit status.
set -eu

: "${CTLOOM_SRC:=/src}"
: "${CTLOOM_WORK:=/work/repo}"
: "${CTLOOM_LIVE:=}"
: "${CTLOOM_DEP_PATH:=}"
: "${CTLOOM_RUNDIR:=/tmp}"

if [ -z "${CTLOOM_BASE:-}" ]; then
    echo "ctloom-warm: CTLOOM_BASE is unset. Pass the commit the clone is to start at" >&2
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

# The settings files the tools read from HOME and XDG_CONFIG_HOME are not read from
# the cache mount: XDG_CONFIG_HOME names a directory that is not there, pip and npm
# read no user configuration, and mvn sources no mavenrc
export XDG_CONFIG_HOME=/nonexistent
export PIP_CONFIG_FILE=/dev/null
export NPM_CONFIG_USERCONFIG=/dev/null
export MAVEN_SKIP_RC=1
# The default test commands and run-tests tell from this that they run to fill the
# package cache
export CTLOOM_FILL=1

# == Step 1: copy the repository to somewhere writable ===
git clone --no-hardlinks --quiet "$CTLOOM_SRC" "$CTLOOM_WORK"

if ! git -C "$CTLOOM_WORK" cat-file -e "$CTLOOM_BASE^{commit}" 2>/dev/null; then
    echo "ctloom-warm: $CTLOOM_BASE is not in the clone of $CTLOOM_SRC. Only a commit some" \
         "branch reaches is copied by git clone" >&2
    exit 2
fi
git -C "$CTLOOM_WORK" checkout --quiet --detach "$CTLOOM_BASE"

# == Step 2: take the dependency files of the running change ==
# A change makes them in its own copy without committing them, so the clone does not
# carry them. The host decides which of them are taken, and reads the same list to
# tell whether the cache already holds them
if [ -n "$CTLOOM_LIVE" ] && [ -d "$CTLOOM_LIVE" ] && [ -n "$CTLOOM_DEP_PATH" ]; then
    printf '%s\n' "$CTLOOM_DEP_PATH" | while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        mkdir -p "$CTLOOM_WORK/$(dirname "$rel")"
        cp "$CTLOOM_LIVE/$rel" "$CTLOOM_WORK/$rel"
        echo "ctloom-warm: took $rel from the running change" >&2
    done
fi

# == Step 3: have ALL_PROXY stand in for the proxy variables that are not set ==
# A tool that reads HTTPS_PROXY and HTTP_PROXY alone is given the proxy ALL_PROXY names
# when that proxy is spoken to over HTTP. A variable set in either case is left as it is
all_proxy_value="${all_proxy:-${ALL_PROXY:-}}"
case "$all_proxy_value" in
    ""|*://*) ;;
    *) all_proxy_value="http://$all_proxy_value" ;;
esac
case "$all_proxy_value" in
    http://*|https://*)
        [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ] || export HTTPS_PROXY="$all_proxy_value"
        [ -n "${http_proxy:-${HTTP_PROXY:-}}" ] || export HTTP_PROXY="$all_proxy_value"
        ;;
esac

# == Step 4: hand the JVM tools the proxy and the certificate authorities ==
# In an image that has ctloom-jvm-route, what it prints is added to JAVA_TOOL_OPTIONS,
# which every JVM started below reads. When it fails, nothing is added
if command -v ctloom-jvm-route >/dev/null 2>&1; then
    jvm_option="$(ctloom-jvm-route "$CTLOOM_RUNDIR")" || jvm_option=""
    if [ -n "$jvm_option" ]; then
        export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }$jvm_option"
    fi
fi

# == Step 5: start Gradle's home from the dependencies the containers share ==
# In an image that has ctloom-gradle-share, the home this container has to itself
# (GRADLE_USER_HOME) is started from the directory the others read fetched dependencies
# from (GRADLE_RO_DEP_CACHE), and Gradle is run without that directory named
gradle_share=""
seed_status=0
if command -v ctloom-gradle-share >/dev/null 2>&1 \
        && [ -n "${GRADLE_RO_DEP_CACHE:-}" ] && [ -n "${GRADLE_USER_HOME:-}" ]; then
    gradle_share="$GRADLE_RO_DEP_CACHE"
    unset GRADLE_RO_DEP_CACHE
    ctloom-gradle-share seed "$gradle_share" "$GRADLE_USER_HOME" || seed_status=$?
fi

# == Step 6: run the toolchain over the clone ============
# Not run from inside the clone, as the sandbox entry point is not
cd "$CTLOOM_RUNDIR"

if [ -z "$gradle_share" ]; then
    exec python -P -m codetwine_loom "$@"
fi

status=0
python -P -m codetwine_loom "$@" || status=$?

# == Step 7: put what Gradle fetched where the containers read it ==
# A home that was not started from that directory is not put there: it holds less than
# the directory does. The run's exit status is kept, unless the run ended with 0 and
# the home could not be started or put there
if [ "$seed_status" -eq 0 ]; then
    ctloom-gradle-share publish "$GRADLE_USER_HOME" "$gradle_share" || seed_status=$?
fi
if [ "$status" -eq 0 ] && [ "$seed_status" -ne 0 ]; then
    status=1
fi
exit "$status"
