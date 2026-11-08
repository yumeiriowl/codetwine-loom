# What ctloom-check-c, ctloom-test-c and ctloom-run-c share, read with ".": reading their
# arguments, configuring the CMake project in the current directory, with the sources
# FetchContent downloads kept in the cache, and running make over a copy of a project
# that has a makefile, or an autotools configure to make one, and no CMakeLists.txt.
#
# A configure that may fetch copies the sources FetchContent downloaded (the non-empty
# _deps/*-src of the build directory) to
# ${XDG_CACHE_HOME:-$HOME/.cache}/ctloom-fetchcontent/<checksum of the repository path>,
# whether or not the rest of the configure went through. An offline configure copies
# them from there into the build directory, over the ones an earlier configure left
# there, and tells FetchContent not to fetch (FETCHCONTENT_FULLY_DISCONNECTED).

# Read the arguments: --offline sets offline to 1 (configure with the FetchContent sources
# kept in the cache and fetch nothing), --no-make sets use_make to 0 (leave a makefile
# alone, as for a language that is not the repository's primary one). Others are passed
# over. Usage: ctloom_read_args "$@"
ctloom_read_args() {
    offline=0
    use_make=1
    for arg in "$@"; do
        case "$arg" in
            --offline) offline=1 ;;
            --no-make) use_make=0 ;;
        esac
    done
}

# Print the directory the FetchContent sources of the current repository are kept in.
ctloom_fetch_cache_dir() {
    printf '%s/ctloom-fetchcontent/%s\n' "${XDG_CACHE_HOME:-$HOME/.cache}" \
        "$(pwd | cksum | cut -d' ' -f1)"
}

# Print the build directory named <prefix>-<checksum of the repository path> under TMPDIR.
ctloom_build_dir() {
    printf '%s/%s-%s\n' "${TMPDIR:-/tmp}" "$1" "$(pwd | cksum | cut -d' ' -f1)"
}

# Copy stdin to stdout with each "<build dir>/" written as "<repository dir>/".
# Usage: ctloom_as_repo_path <build dir> <repository dir>
ctloom_as_repo_path() {
    sed "s|$1/|$2/|g"
}

# Exit with <status> when one of the named tools is not installed, naming it on stderr.
# Usage: ctloom_need_tool <script name> <status> <tool>...
ctloom_need_tool() {
    need_name=$1
    need_status=$2
    shift 2
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "$need_name: $tool does not exist" >&2
            exit "$need_status"
        fi
    done
}

# Configure the project into a build directory with Ninja. cmake's output is left in
# configure_out, and 1 is returned when cmake failed.
# Usage: ctloom_configure <build dir> <offline: 0 or 1> [cmake argument]...
ctloom_configure() {
    configure_dir=$1
    configure_offline=$2
    shift 2
    fetch_dir=$(ctloom_fetch_cache_dir)
    if [ "$configure_offline" = 1 ]; then
        if [ -d "$fetch_dir" ]; then
            mkdir -p "$configure_dir/_deps"
            for src in "$fetch_dir"/*-src; do
                [ -d "$src" ] || continue
                rm -rf "$configure_dir/_deps/${src##*/}"
                cp -a "$src" "$configure_dir/_deps/"
            done
        fi
        set -- "$@" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    fi
    configure_status=0
    configure_out=$(cmake -G Ninja -S . -B "$configure_dir" "$@" 2>&1) || configure_status=1
    if [ "$configure_offline" = 0 ]; then
        for src in "$configure_dir"/_deps/*-src; do
            [ -n "$(ls -A "$src" 2>/dev/null)" ] || continue
            mkdir -p "$fetch_dir"
            rm -rf "$fetch_dir/${src##*/}.new"
            cp -a "$src" "$fetch_dir/${src##*/}.new" \
                && rm -rf "$fetch_dir/${src##*/}" \
                && mv "$fetch_dir/${src##*/}.new" "$fetch_dir/${src##*/}"
        done
    fi
    return "$configure_status"
}

# Print the makefile GNU make reads in the current directory (GNUmakefile, makefile or
# Makefile, the first that exists), or nothing when there is none.
ctloom_makefile() {
    for makefile_name in GNUmakefile makefile Makefile; do
        if [ -f "$makefile_name" ]; then
            printf '%s\n' "$makefile_name"
            return 0
        fi
    done
}

# Print the makefile of the current directory (ctloom_makefile), or "configure" when there
# is none and there is a configure script, configure.ac or configure.in to make one
# (ctloom_autotools_configure), or nothing. Prints nothing when use_make is 0
# (ctloom_read_args).
ctloom_make_source() {
    [ "${use_make:-1}" = 0 ] && return 0
    makefile_name=$(ctloom_makefile)
    if [ -n "$makefile_name" ]; then
        printf '%s\n' "$makefile_name"
    elif [ -f configure ] || [ -f configure.ac ] || [ -f configure.in ]; then
        printf 'configure\n'
    fi
}

# Make the makefile of the autotools project copied to build_dir: "autoreconf -fi" when
# there is no configure script, then "./configure", with the build directory's path in
# their output written as the repository's. Exits with <status>, the output on stderr,
# when autoreconf is not installed or a step fails.
# Usage: ctloom_autotools_configure <script name> <status> <repository dir>
ctloom_autotools_configure() {
    if [ ! -f "$build_dir/configure" ]; then
        ctloom_need_tool "$1" "$2" autoreconf
        if ! step_out=$(cd "$build_dir" && autoreconf -fi 2>&1); then
            echo "$1: autoreconf -fi failed:" >&2
            printf '%s\n' "$step_out" | ctloom_as_repo_path "$build_dir" "$3" >&2
            exit "$2"
        fi
    fi
    if ! step_out=$(cd "$build_dir" && sh ./configure 2>&1); then
        echo "$1: ./configure failed:" >&2
        printf '%s\n' "$step_out" | ctloom_as_repo_path "$build_dir" "$3" | tail -n 40 >&2
        exit "$2"
    fi
}

# Make the build directory <script name>-make-<checksum of the repository path> under
# TMPDIR, named in build_dir, and copy the current directory, less .git, into it after
# emptying it. A copy with no makefile gets one from ctloom_autotools_configure. The
# copy's makefile (ctloom_makefile) is named in makefile. Exits with <status> when make
# is not installed, the copy failed or the makefile cannot be made, naming why on stderr.
# Usage: ctloom_make_setup <script name> <status>
ctloom_make_setup() {
    ctloom_need_tool "$1" "$2" make
    build_dir=$(ctloom_build_dir "$1-make")
    if ! { rm -rf "$build_dir" && mkdir -p "$build_dir"; } \
            || ! tar -cf - --exclude=./.git . | tar -xf - -C "$build_dir"; then
        echo "$1: $(pwd) could not be copied to $build_dir" >&2
        exit "$2"
    fi
    makefile=$(cd "$build_dir" && ctloom_makefile)
    if [ -z "$makefile" ]; then
        ctloom_autotools_configure "$1" "$2" "$(pwd)"
        makefile=$(cd "$build_dir" && ctloom_makefile)
    fi
}

# Print the first of the named targets the makefile in the current directory has a rule
# for, or nothing when it has none of them. make's built-in rules are left out (a test.c
# there is not a test target), and a target that names a file or directory there and that
# make has nothing to do for (a test/ directory with no test rule) is not one.
# Usage: ctloom_make_target <target>...
ctloom_make_target() {
    for make_target in "$@"; do
        case "$(LC_ALL=C make -n -r "$make_target" 2>&1)" in
            *"No rule to make target '$make_target'"*) ;;
            *"Nothing to be done for '$make_target'"*)
                [ -e "$make_target" ] || { printf '%s\n' "$make_target"; return 0; } ;;
            *) printf '%s\n' "$make_target"
               return 0 ;;
        esac
    done
}

# Run make with the arguments in the build directory and leave what it printed in
# make_out, with the build directory's path written as the repository's, and its status
# in make_status, which is also returned. A build directory with a Makefile.am is made
# with as many jobs at once as CMAKE_BUILD_PARALLEL_LEVEL says (1 without it), any other
# with make's default.
# Usage: ctloom_make_in <build dir> <repository dir> <make argument>...
ctloom_make_in() {
    make_dir=$1
    make_repo=$2
    shift 2
    make_status=0
    if [ -f "$make_dir/Makefile.am" ]; then
        make_out=$(cd "$make_dir" && make -j"${CMAKE_BUILD_PARALLEL_LEVEL:-1}" "$@" 2>&1) \
            || make_status=$?
    else
        make_out=$(cd "$make_dir" && make "$@" 2>&1) || make_status=$?
    fi
    make_out=$(printf '%s\n' "$make_out" | ctloom_as_repo_path "$make_dir" "$make_repo")
    return "$make_status"
}

# Print the lines of make's output that say a command is not installed, and return 0,
# when every step that failed is explained by them: a step ending with Error 127 right
# after the shell's "<sh>: <n>: <command>: not found" (dash) or "<sh>: line <n>:
# <command>: command not found" (bash) or make's "<command>: No such file or directory"
# for a command named without "/", and a step ending with Error 2 right after a failed
# step of a make below it. Returns 1, printing nothing, when anything
# else failed: another step, a compiler or linker error, a prerequisite with no rule, or
# make's own error in a makefile ("<file>:<line>: *** <message>").
# Usage: ctloom_make_not_installed <make's output>
ctloom_make_not_installed() {
    printf '%s\n' "$1" | awk '
        function depth_of(line) {
            return match(line, /^make\[[0-9]+\]/) ? substr(line, 6, RLENGTH - 6) + 0 : 0
        }
        /sh: [0-9]+: [^\/: ]+: not found$/ \
                || /: line [0-9]+: [^\/: ]+: command not found$/ \
                || /^make(\[[0-9]+\])?: [^\/: ]+: No such file or directory$/ {
            found = found $0 "\n"
            not_found = 1
            next
        }
        /^make(\[[0-9]+\])?: \*\*\* \[/ {
            depth = depth_of($0)
            if (!(/\] Error 127$/ && not_found) && !(/\] Error 2$/ && last_depth > depth))
                other = 1
            last_depth = depth
            not_found = 0
            next
        }
        / (fatal )?error:|cannot find -l|undefined reference/ { other = 1 }
        /\*\*\* No rule to make target |^[^: ]+:[0-9]+: \*\*\* / { other = 1 }
        END {
            if (found == "" || other) exit 1
            printf "%s", found
        }'
}

# Print the lines of make's output the static check counts: the compiler's error lines,
# less a missing header ("fatal error: <h>: No such file or directory") that is neither a
# file of the current directory nor one of git's HEAD there, a prerequisite make has no
# rule for, as "<makefile>:1: error: no rule to make target '<path>', needed by
# '<target>'.", and make's own error in a makefile ("<file>:<line>: *** <message>") as
# "<file>:<line>: error: <message>".
# Usage: ctloom_make_check_lines <makefile> <make's output>
ctloom_make_check_lines() {
    head_files=$(git ls-tree -r --name-only HEAD 2>/dev/null)
    absent=$(printf '%s\n' "$2" \
        | sed -n 's/.*fatal error: \(.*\): No such file or directory$/\1/p' | sort -u \
        | while IFS= read -r header; do
            name=$header
            while :; do
                case "$name" in
                    ./*|../*) name=${name#*/} ;;
                    *) break ;;
                esac
            done
            [ -n "$(find . -path "*/$name" -print 2>/dev/null | head -n 1)" ] \
                || printf '%s\n' "$head_files" | awk -v name="$name" '
                    $0 == name || substr($0, length($0) - length(name)) == "/" name {
                        found = 1
                    }
                    END { exit !found }' \
                || printf '%s\n' "$header"
        done)
    printf '%s\n' "$2" | ABSENT="$absent" MAKEFILE="$1" awk '
        BEGIN {
            count = split(ENVIRON["ABSENT"], header_list, "\n")
            for (i = 1; i <= count; i++)
                absent["fatal error: " header_list[i] ": No such file or directory"] = 1
        }
        /\*\*\* No rule to make target .*, needed by / {
            sub(/^make(\[[0-9]+\])?: \*\*\* No rule to make target /, "")
            print ENVIRON["MAKEFILE"] ":1: error: no rule to make target " $0
            next
        }
        /^[^: ]+:[0-9]+: \*\*\* / {
            sub(/: \*\*\* /, ": error: ")
            print
            next
        }
        / (fatal )?error:/ {
            for (line in absent) if (index($0, line)) next
            print
        }'
}
