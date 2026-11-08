"""The per-language settings table.

One LangConfig per extension. This module imports nothing from config.py.
"""

import fnmatch
import re
import sys
from collections.abc import Callable, Set
from dataclasses import dataclass

from codetwine.utils.file_utils import copy_path_to_rel

# == Stopwords ===============================================
# English function words, the same for each language
STOPWORD_BASE_SET = frozenset("""
the and for are but not you all can has have had was were this that these those
with from into onto over under then than when where which while what who its his her
their them there here been being does did done will would should could may might must
shall also such some any each every other more most much many few one two both out off
per via use used using only just how why yes no
to of by or an on it be at as in is if do up so
""".split())


@dataclass(frozen=True)
class LangConfig:
    """Everything configured for one language (one extension)."""

    exclude_type_set: frozenset[str]  # definition types kept out of the search
    stopword_set: frozenset[str]      # this language's keywords and type names
    # The command run_test runs. Empty means this language has no default test command:
    # for this language as the primary one, with TEST_COMMAND not set, run_test is not
    # handed to the model when no other language's tests can be run either
    test_command_tuple: tuple[str, ...]
    # Where the tests live, relative to the repository root. The same directory under
    # any other directory holds tests too
    test_path: str
    test_target: str              # how paths are passed: "path" / "separator" / "none"
    # The other directories the tests are kept in, relative to the repository root. Of
    # test_path and these, the one in the repository holding the most source files is
    # taken (config.test_dir_of). Empty means test_path alone
    test_path_alt_tuple: tuple[str, ...] = ()
    # The command run_test runs in place of test_command_tuple for this language as the
    # primary one, with TEST_COMMAND not set, while the repository holds no file of
    # dep_file_tuple outside the directories build_output_tuple names by name alone. The
    # test files go after it as paths. Empty means test_command_tuple runs whatever the
    # repository holds
    bare_test_command_tuple: tuple[str, ...] = ()
    # The names of a directory or a file that bare_test_command_tuple takes for a test
    # wherever they stand in a path when it is given no paths, as fnmatch patterns
    # matched case-sensitively against each part of it
    bare_test_name_tuple: tuple[str, ...] = ()
    # What run_project puts in front of the model's args. Empty means this language has
    # no default launcher, and RUN_COMMAND has to be set for the tool to run anything
    run_command_tuple: tuple[str, ...] = ()
    # The names of the files one of which the repository's top directory holds for
    # run_command_tuple to be the default. Empty means it is the default wherever
    run_file_tuple: tuple[str, ...] = ()
    # The names of the files besides dep_file_tuple, root_target_tuple,
    # test_build_file_tuple and run_file_tuple that make a directory one of this
    # language's projects, as fnmatch patterns matched against a file's name
    project_file_tuple: tuple[str, ...] = ()
    # Whether this language is passed over as the primary one when the repository's top
    # directory holds no file of its project_file_pattern_tuple() and does hold a
    # project file of another language counted (index.store.primary_ext)
    need_root_project: bool = False
    # What the model is told of run_command_tuple where it is the launcher in force, one
    # sentence or more. Empty says nothing more than the command's name
    run_note: str = ""
    # The names of a directory or a file that mark a test wherever they stand in a path,
    # as fnmatch patterns matched case-sensitively against each part of it
    # ("__tests__", "*.test.[jt]s"). Empty means only test_path marks the tests
    test_name_tuple: tuple[str, ...] = ()
    # The file names that say what this language's build tool has to fetch, as fnmatch
    # patterns ("*.csproj") matched against the base name of a path; a name without "*",
    # "?" or "[" matches itself only. Empty means this language has no such file
    dep_file_tuple: tuple[str, ...] = ()
    # What tells this language's build tool to work from the package cache alone, as
    # (variable, value) pairs. Empty means the tool fetches nothing
    offline_env_tuple: tuple[tuple[str, str], ...] = ()
    # The same thing for a tool that has no setting of its own: what is added to the
    # default test command and static check when config.TOOL_OFFLINE says the cache is
    # all there is. It goes at the end of the command
    offline_arg_tuple: tuple[str, ...] = ()
    # What is added to the default test command and static check when this language is
    # not the repository's primary one. It goes at the end of the command
    other_arg_tuple: tuple[str, ...] = ()
    # The files besides dep_file_tuple that say which tests are built and run, which the
    # stage that writes tests may write as well when this is the primary language
    test_build_file_tuple: tuple[str, ...] = ()
    # What puts the dependencies this language's test command loads where it reads them,
    # run before that command in a container this tool started, from the directory the
    # test command is started in. Empty means the test command fetches what it needs
    # itself. The ctloom-install-* commands are the scripts under sandbox/
    install_command_tuple: tuple[str, ...] = ()
    # The static check, run from the repository root. Each output line of the form
    # "<path>:<line>[:<column>]: <message>" or "<path>(<line>,<column>): <message>" is
    # one finding. Empty means this language has no default check, and CHECK_COMMAND has
    # to be set for one to run. The ctloom-check-* commands are the scripts under
    # sandbox/
    check_command_tuple: tuple[str, ...] = ()
    # Whether two names of this language are the same name whatever their upper and
    # lower case
    ignore_case: bool = False
    # Whether a name of this language can join words with "-" (CUST-ACCT-ID)
    hyphen_name: bool = False
    # Whether a line of this language can be in fixed format: a sequence area in columns
    # 1-6, an indicator in column 7, the code in columns 8-72 and an identification area
    # after it
    column_layout: bool = False
    # The definition type of a data item whose items are nested by line range. "" when
    # the language has none
    record_type: str = ""
    # What the stages that write code are told about this language, as pieces of the
    # instructions that follow one another, the last ending in a newline: (the tool the
    # piece is for, or "" for a piece given whatever the tools; the piece). Empty adds
    # nothing
    edit_note_tuple: tuple[tuple[str, str], ...] = ()
    # What this language's tools (its test command, static check, launcher and build
    # commands) write into the repository, as .gitignore patterns ("target/" is a
    # directory of that name at any depth, "/dist/" one in a directory holding one of
    # the language's dependency files, "uv.lock" a file of that name at any depth)
    build_output_tuple: tuple[str, ...] = ()
    # What the test command and the static check are pointed at, as groups of fnmatch
    # patterns matched against the file names in the repository root, in order. The
    # first group that matches a file decides: when it matches one file, that name goes
    # after the language's default test_command_tuple and check_command_tuple; when it
    # matches more, nothing does. Empty adds nothing
    root_target_tuple: tuple[tuple[str, ...], ...] = ()
    # A line that makes a source file of this language fail to build when it is added at
    # the end. "" names none
    compile_error_line: str = ""
    # A regex for the line the test command prints for each test file it runs, with
    # {name} standing for the file's base name without its extension. "" when the
    # command prints no such line
    run_line_pattern: str = ""
    # The definition types, among exclude_type_set's, that give a variable, a constant
    # or a field of a type a value
    value_type_set: frozenset[str] = frozenset()
    # A regex for a function or a class written as a value (a lambda, an arrow function,
    # a class expression). "" when a value of the language holds none
    function_value_pattern: str = ""
    # What a comment of the language is written between: what starts one that runs to
    # the end of its line, then what starts and what ends one that runs over lines. ()
    # names none
    comment_mark_tuple: tuple[str, ...] = ()
    # A regex for the front of a definition's name that is no part of the name the
    # definition is told by: the names of the definitions it belongs to, where one
    # written outside them carries their names in its own ("Calc::" in "Calc::mul"), and
    # the mark that makes a name out of its type's ("~" in "~Calc"). "" when the names
    # of the language carry none
    name_prefix_pattern: str = ""
    # The name and the text of the test file a filling of the package cache writes in
    # the tests' directory of each directory holding one of module_file_tuple's files,
    # when no file of the language is under the parent of any of those tests'
    # directories. () writes none
    fill_test_tuple: tuple[str, str] | tuple[()] = ()
    # The names of the build files that make a directory one whose tests the test
    # command builds
    module_file_tuple: tuple[str, ...] = ()


# The requirement files a Python directory with no pyproject.toml, or with one that has
# no [project] table, is run with, each where it is there (tools.edit.uv_project_word_list)
PY_REQUIREMENT_FILE_TUPLE = ("requirements.txt", "requirements-dev.txt",
                             "requirements_dev.txt", "requirements-test.txt",
                             "requirements_test.txt", "dev-requirements.txt",
                             "test-requirements.txt")

_PY = LangConfig(
    exclude_type_set=frozenset({"expression_statement"}),
    stopword_set=frozenset("""
    str dict list int float bool none true false self cls def class import return
    yield pass raise try except finally lambda async await as in is if else elif
    tuple set bytes object type optional union callable iterable sequence mapping
    """.split()),
    test_command_tuple=("uv", "run", "--with", "pytest", "--with", "pytest-asyncio",
                        "pytest", "--asyncio-mode=auto", "-q",
                        "-W", "ignore:The configuration option "
                        "\"asyncio_default_fixture_loop_scope\" is unset."
                        ":pytest.PytestDeprecationWarning"),
    test_path="tests", test_target="path", test_path_alt_tuple=("test", "testing"),
    run_command_tuple=("uv", "run"),
    dep_file_tuple=("pyproject.toml", *PY_REQUIREMENT_FILE_TUPLE, "setup.py", "setup.cfg",
                    "uv.lock", "Pipfile", "poetry.lock"),
    offline_env_tuple=(("UV_OFFLINE", "1"),),
    build_output_tuple=("__pycache__/", "uv.lock", "*.egg-info/", "/dist/", "/build/"),
    compile_error_line=")",
    value_type_set=frozenset({"expression_statement"}),
    function_value_pattern=r"\blambda\b",
    check_command_tuple=(sys.executable, "-m", "ruff", "check", "--select", "F821",
                         "--output-format", "concise", "--no-cache", "--exit-zero", "."))

# How a comment is written in the languages that write one as C does
_C_COMMENT_MARK_TUPLE = ("//", "/*", "*/")

_JAVA = LangConfig(
    project_file_tuple=("build.gradle", "build.gradle.kts", "settings.gradle",
                        "settings.gradle.kts"),
    need_root_project=True,
    exclude_type_set=frozenset({"field_declaration", "constant_declaration",
                                "enum_constant"}),
    stopword_set=frozenset("""
    public private protected static final void string boolean char long short
    byte double float int new extends implements throws interface abstract
    class import package return null true false this super enum
    """.split()),
    # Without -q, the output keeps the "Tests run: 2, Failures: 0" and "No tests to
    # run." lines. With -fae, the modules that do not depend on a failing module run
    # their tests
    test_command_tuple=("mvn", "-B", "-fae", "test"),
    test_path="src/test/java", test_target="none",
    dep_file_tuple=("pom.xml",),
    build_output_tuple=("target/",),
    # mvn takes the offline flag from MAVEN_ARGS
    offline_env_tuple=(("MAVEN_ARGS", "-o"),),
    compile_error_line="#error",
    run_line_pattern=r"^(?:\[INFO\] )?Running (?:[\w$]+\.)*{name}$",
    value_type_set=frozenset({"field_declaration"}),
    # A lambda, a method reference, the body of an anonymous class
    function_value_pattern=r"->|::|\)\s*\{",
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE,
    fill_test_tuple=("CtloomFillTest.java", "class CtloomFillTest {\n}\n"),
    module_file_tuple=("pom.xml",))

_KOTLIN = LangConfig(
    need_root_project=True,
    exclude_type_set=frozenset({"property_declaration", "enum_entry"}),
    stopword_set=frozenset("""
    fun val var class object interface import package return null true false
    this super override private public internal protected suspend data sealed
    companion when is as in out string int long boolean unit any
    """.split()),
    # Without -q, the output keeps the "FooTest > bar() FAILED" lines. With --continue,
    # the tasks that do not depend on a failing task run
    test_command_tuple=("gradle", "test", "--continue", "--console=plain"),
    test_path="src/test/kotlin", test_target="none",
    dep_file_tuple=("build.gradle", "build.gradle.kts", "settings.gradle",
                    "settings.gradle.kts", "libs.versions.toml"),
    build_output_tuple=("build/", ".gradle/", ".kotlin/"),
    # The offline flag goes on gradle's command line
    offline_arg_tuple=("--offline",),
    compile_error_line="#error",
    value_type_set=frozenset({"property_declaration"}),
    # A lambda or a body, an anonymous function, a function type, a function
    # reference, or an accessor: "get(" or "set(" that is no call of a member
    function_value_pattern=r"[{]|\bfun\b|->|::|(?<![.\w])(?:get|set)\s*\(",
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE,
    fill_test_tuple=("CtloomFillTest.kt", "class CtloomFillTest\n"),
    module_file_tuple=("build.gradle", "build.gradle.kts"))

_JS = LangConfig(
    need_root_project=True,
    exclude_type_set=frozenset({"lexical_declaration", "variable_declaration",
                                "field_definition"}),
    stopword_set=frozenset("""
    const let var function return class extends import export default from
    new this null undefined true false async await typeof instanceof
    """.split()),
    test_command_tuple=("npm", "test", "--silent"),
    bare_test_command_tuple=("node", "--test"),
    bare_test_name_tuple=("test", "test.js", "test.[cm]js", "test-*.js", "test-*.[cm]js",
                          "*.test.js", "*.test.[cm]js", "*-test.js", "*-test.[cm]js",
                          "*_test.js", "*_test.[cm]js"),
    test_path="test", test_target="separator", test_path_alt_tuple=("tests",),
    test_name_tuple=("__tests__", "*.test.[jt]s", "*.test.[jt]sx", "*.test.[cm][jt]s",
                     "*.spec.[jt]s", "*.spec.[jt]sx", "*.spec.[cm][jt]s"),
    dep_file_tuple=("package.json", "package-lock.json", "yarn.lock",
                    "pnpm-lock.yaml", "pnpm-workspace.yaml", "bun.lock", "bun.lockb",
                    "bunfig.toml"),
    # npm takes the offline setting from npm_config_offline
    offline_env_tuple=(("npm_config_offline", "true"),),
    install_command_tuple=("ctloom-install-js",),
    build_output_tuple=("node_modules/", ".yarn/install-state.gz", "/dist/", "/build/",
                        "/out/", "/coverage/", "/.output/", ".next/", ".nuxt/", ".svelte-kit/",
                        ".angular/", ".turbo/", ".parcel-cache/", ".docusaurus/", ".astro/",
                        "*.tsbuildinfo"),
    check_command_tuple=("ctloom-check-js",),
    value_type_set=frozenset({"lexical_declaration", "variable_declaration",
                              "field_definition"}),
    # An arrow function, a function or class expression, a method of an object literal
    function_value_pattern=r"=>|\bfunction\b|\bclass\b|\)\s*\{",
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE)

_TS = LangConfig(
    need_root_project=True,
    exclude_type_set=frozenset({"lexical_declaration", "variable_declaration",
                                "public_field_definition", "property_signature",
                                "enum_assignment", "property_identifier"}),
    stopword_set=_JS.stopword_set | frozenset(
        "interface type enum implements readonly public private protected "
        "string number boolean any unknown never void".split()),
    test_command_tuple=_JS.test_command_tuple, test_path="test", test_target="separator",
    test_path_alt_tuple=_JS.test_path_alt_tuple, test_name_tuple=_JS.test_name_tuple,
    dep_file_tuple=_JS.dep_file_tuple,
    offline_env_tuple=_JS.offline_env_tuple,
    install_command_tuple=_JS.install_command_tuple,
    build_output_tuple=_JS.build_output_tuple,
    check_command_tuple=("ctloom-check-ts",),
    value_type_set=frozenset({"lexical_declaration", "variable_declaration",
                              "public_field_definition"}),
    # JavaScript's, and a method of an object literal with a return type written
    # between its parameters and its body
    function_value_pattern=_JS.function_value_pattern + r"|\)\s*:[^=;)]*\{",
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE)

# The makefile names GNU make reads, in the order it looks for them
_MAKEFILE_TUPLE = ("GNUmakefile", "makefile", "Makefile")

_C = LangConfig(
    need_root_project=True,
    exclude_type_set=frozenset({"declaration", "preproc_def", "field_declaration",
                                "enumerator"}),
    stopword_set=frozenset("""
    int char void long short unsigned signed float double struct union enum
    typedef static const extern return sizeof include define ifdef endif
    """.split()),
    test_command_tuple=("ctloom-test-c",),
    test_path="tests", test_target="none", test_path_alt_tuple=("test",),
    dep_file_tuple=("CMakeLists.txt",),
    offline_arg_tuple=("--offline",),
    other_arg_tuple=("--no-make",),
    test_build_file_tuple=(*_MAKEFILE_TUPLE, "Makefile.am"),
    check_command_tuple=("ctloom-check-c",),
    run_command_tuple=("ctloom-run-c",),
    run_file_tuple=(*_MAKEFILE_TUPLE, "configure", "configure.ac", "configure.in"),
    compile_error_line="#error",
    value_type_set=frozenset({"declaration", "preproc_def"}),
    edit_note_tuple=(
        ("", "## C and C++ tests\n\n- "),
        ("run_test", "run_test() configures and builds the CMake project and runs its "
                     "CTest tests, or, with no CMakeLists.txt, runs make's test target "
                     "(check when there is none). "),
        ("", """A test
  runs only when CMakeLists.txt builds its source file (add_executable) and registers
  it (add_test); a test file CMakeLists.txt does not name is never built or run. Add a
  test to a test file CMakeLists.txt already names, or add both lines for a new one.
  In a repository with a makefile and no CMakeLists.txt at the top, a test runs only
  when the makefile's test target builds and runs it: add a test to a test file it
  already builds, or have that target build and run a new one, and do not add a
  CMakeLists.txt. Where the repository has a Makefile.am, make check runs the tests its
  TESTS names (check_PROGRAMS builds them): add a new test there, and do not write a
  Makefile.

""")),
    run_note=("ctloom-run-c runs make in a copy of the repository made for each call, so "
              "your args are make's targets and variables, not a program's path: "
              "run_project([\"<target>\"]) is make <target>, and a file of file_dict= "
              "reaches the makefile as NAME={workspace}/<path>."))

_CPP = LangConfig(
    need_root_project=True,
    exclude_type_set=_C.exclude_type_set | {"namespace_definition"},
    stopword_set=_C.stopword_set | frozenset(
        "class public private protected virtual template namespace using new "
        "delete this nullptr bool true false operator friend inline".split()),
    test_command_tuple=_C.test_command_tuple, test_path="tests", test_target="none",
    test_path_alt_tuple=_C.test_path_alt_tuple,
    dep_file_tuple=_C.dep_file_tuple, offline_arg_tuple=_C.offline_arg_tuple,
    other_arg_tuple=_C.other_arg_tuple, test_build_file_tuple=_C.test_build_file_tuple,
    check_command_tuple=_C.check_command_tuple, edit_note_tuple=_C.edit_note_tuple,
    run_command_tuple=_C.run_command_tuple, run_file_tuple=_C.run_file_tuple,
    run_note=_C.run_note,
    compile_error_line=_C.compile_error_line, value_type_set=_C.value_type_set,
    # A lambda: its capture list, then its template parameters, its parameters, its
    # body, what stands between the list and the body where there are no parameters,
    # or a name with "(" after it (an attribute, an exception specification, a macro)
    function_value_pattern=(r"\]\s*(?:[({<]|->|\b(?:mutable|constexpr|consteval|static"
                            r"|noexcept|requires)\b|[A-Za-z_]\w*\s*\()"),
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE,
    name_prefix_pattern=r"^(?:(?!operator\b)[^:\s]+::)*~?")

_RS = LangConfig(
    need_root_project=True,
    exclude_type_set=frozenset({"const_item", "static_item", "mod_item",
                                "associated_type"}),
    stopword_set=frozenset("""
    fn let mut pub struct enum impl trait use mod crate self super match if else loop
    while for in return where as ref move async await dyn const static type unsafe
    extern true false some none ok err string vec option result box bool str char
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64
    """.split()),
    test_command_tuple=("cargo", "test", "--workspace", "--no-fail-fast"),
    test_path="tests", test_target="none",
    run_command_tuple=("cargo", "run", "--quiet", "--"),
    dep_file_tuple=("Cargo.toml", "Cargo.lock"),
    offline_env_tuple=(("CARGO_NET_OFFLINE", "true"),),
    build_output_tuple=("target/",),
    check_command_tuple=("ctloom-check-rs",),
    compile_error_line="#error",
    value_type_set=frozenset({"const_item", "static_item"}),
    # A function pointer type, a closure trait, an impl block, or a closure: a "|"
    # where an expression starts, which is after anything but a name, a literal, a
    # closing bracket or another "|", and after a keyword
    function_value_pattern=(r"\b(?:fn|Fn|FnMut|FnOnce|impl)\b"
                            r"|(?:[^\w\s)\]}\"'?|]"
                            r"|\b(?:move|async|return|break|else|in|mut)\b)\s*\|"),
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE)

# SQL has no test runner of its own: tests run only when TEST_COMMAND is set
_SQL = LangConfig(
    exclude_type_set=frozenset(),
    stopword_set=frozenset("""
    select from where insert into values update set delete create alter drop table
    view index function procedure trigger sequence schema type as on join left right
    inner outer group by order having limit offset and or not null is in exists
    primary key references default unique constraint begin end return returns
    language if then else case when int integer text varchar boolean numeric serial
    """.split()),
    test_command_tuple=(), test_path="tests", test_target="none")

# COBOL has no test runner and no launcher of its own: tests run only when TEST_COMMAND
# is set, and run_project only when RUN_COMMAND is
_CBL = LangConfig(
    exclude_type_set=frozenset({"occurs_indexed"}),
    stopword_set=frozenset("""
    identification environment procedure division section program-id working-storage
    local-storage linkage fd sd using returning entry pic picture value values
    redefines renames occurs times depending indexed ascending descending usage comp
    comp-1 comp-2 comp-3 comp-4 comp-5 computational computational-1 computational-2
    computational-3 computational-4 computational-5 binary packed-decimal national
    pointer filler space spaces zero zeros zeroes low-value low-values high-value
    high-values quote quotes all is are leading trailing separate character sync
    synchronized justified just blank external global recording contains characters
    records label standard omitted varying end
    """.split()),
    test_command_tuple=(), test_path="tests", test_target="none",
    check_command_tuple=("ctloom-check-cbl",),
    ignore_case=True, hyphen_name=True, column_layout=True, record_type="data_description",
    edit_note_tuple=(("", """## COBOL source

- A name is the same name in upper and lower case. read_definition, graph_search and
  grep_source match a COBOL name in either case. Write a new name in the case the
  lines around it use.
- A fixed-format line holds a sequence number in columns 1-6, the indicator in column
  7 ("*" for a comment, "-" for a continuation), area A in columns 8-11 (headers,
  level numbers 01 and 77, FD), area B in columns 12-72 and an identification area
  from column 73 on, which the compiler ignores. Keep every line you write to that
  layout: fill columns 1-6 the way the lines around it do, and continue a statement
  on the next line rather than writing past column 72. A file whose lines do not
  keep to it is in free format; write it the way that file is written.
- The numbers in columns 1-6 are sequence numbers, not line numbers. The tools take
  the line numbers read_source_file's header and read_definition's "lines" give.
- A copybook is copied into every program that names it in a COPY statement. Before
  you change one, find those programs with grep_source and check each of them.

"""),))

# A BMS source (the CICS screen maps). codetwine gives its definitions as the symbolic
# map the source makes: one group per map and input and output (<map>I, <map>O) and the
# field names under them. The lines are in the layout of assembler macros, a label from
# column 1, not in COBOL's fixed format. It has no test runner, no launcher and no check
# of its own
_BMS = LangConfig(
    exclude_type_set=frozenset(),
    stopword_set=frozenset("""
    dfhmsd dfhmdi dfhmdf type map mapset final mode in out inout lang cobol ctrl freekb
    frset alarm print storage auto tioapfx yes no size pos length attrb askip prot unprot
    num brt norm drk ic fset initial picin picout occurs grpname color hilight validn ps
    dsatts mapatts extatt justify line column end
    """.split()),
    test_command_tuple=(), test_path="tests", test_target="none",
    ignore_case=True, record_type="data_description")

# JCL: jobs, procedures and INCLUDE members. codetwine gives the jobs, procedures, named
# steps and symbolic parameters as definitions; a symbolic parameter is a value and is
# left out of the search. It has no test runner, no launcher and no check of its own
_JCL = LangConfig(
    exclude_type_set=frozenset({"symbolic_parameter"}),
    stopword_set=frozenset("""
    job exec pgm proc pend set include jcllib order else endif output cntl endcntl
    parm cond region time class msgclass msglevel notify typrun disp dsn dsname sysout
    space unit vol volume ser dcb recfm lrecl blksize dsorg trk cyl new old shr mod
    catlg uncatlg delete keep pass dummy sysin sysprint systsin systsprt
    """.split()),
    test_command_tuple=(), test_path="tests", test_target="none",
    ignore_case=True)

# CSD: the CICS resource definitions DFHCSDUP reads. codetwine gives each resource a
# DEFINE command defines as a definition (define_transaction, define_program, ...). It
# has no test runner, no launcher and no check of its own
_CSD = LangConfig(
    exclude_type_set=frozenset(),
    stopword_set=frozenset("""
    define userdefine group list add alter delete description status enabled disabled
    transaction program mapset profile file tdqueue tranclass partitionset
    language le370 cobol assembler pli concurrency quasirent threadsafe execkey user cics
    datalocation below taskdataloc taskdatakey storageclear runaway shutdown isolate
    dynamic routable priority dtimout restart spurge tpurge dump trace confdata
    otstimeout action backout wait waittime ressec cmdsec resident usage uselpacopy cedf
    definetime changetime changeusrid changeagent changeagrel csdapi csdbatch
    """.split()),
    test_command_tuple=(), test_path="tests", test_target="none",
    ignore_case=True)

_CS = LangConfig(
    need_root_project=True,
    exclude_type_set=frozenset({"field_declaration", "event_field_declaration",
                                "enum_member_declaration", "property_declaration",
                                "indexer_declaration"}),
    stopword_set=frozenset("""
    abstract as base bool byte char checked class const decimal default delegate do
    double else enum event explicit extern false finally fixed float foreach goto
    implicit in int interface internal is lock long namespace new null object operator
    out override params private protected public readonly ref return sbyte sealed short
    sizeof stackalloc static string struct switch this throw true try typeof uint ulong
    unchecked unsafe ushort using var virtual void volatile while async await get set
    init value record partial where yield nameof dynamic task list
    """.split()),
    test_command_tuple=("dotnet", "test"),
    test_path="tests", test_target="none", test_path_alt_tuple=("test",),
    test_name_tuple=("*.*Tests", "*.Test"),
    dep_file_tuple=("*.csproj", "Directory.Build.props", "Directory.Packages.props",
                    "packages.lock.json", "global.json", "NuGet.Config", "nuget.config"),
    offline_env_tuple=(("NuGetAudit", "false"),),
    check_command_tuple=("ctloom-check-cs",),
    build_output_tuple=("bin/", "obj/"),
    root_target_tuple=(("*.sln", "*.slnx"), ("*.csproj",)),
    compile_error_line="#error",
    name_prefix_pattern=r"^~",
    value_type_set=frozenset({"field_declaration", "property_declaration"}),
    # A lambda or an expression body, an anonymous method, an accessor with a body
    function_value_pattern=r"=>|\bdelegate\b|\b(?:get|set|init|add|remove)\s*\{",
    comment_mark_tuple=_C_COMMENT_MARK_TUPLE,
    edit_note_tuple=(("", """## C# source

- A project (.csproj) builds every .cs file under its own directory. Put a new source
  file under the directory of the project that uses it, and a new test file under the
  directory of an existing test project (one that references xUnit, NUnit or MSTest).
  A .cs file under no project is not built, and a test in it never runs.
- A partial type is declared in more than one file. Before you add or change a member
  of one, read every file read_definition gives for the type.

"""),))

# R scripts, and the R code chunks of R Markdown and Quarto documents. ctloom-test-r
# runs the testthat tests of a package (a DESCRIPTION file at the top) or of the scripts
# under tests/testthat, and takes test files
_R = LangConfig(
    exclude_type_set=frozenset({"binary_operator"}),
    stopword_set=frozenset("""
    function if else for while repeat break next return in true false null na inf nan
    library require source invisible stop warning message list vector character numeric
    integer logical data frame
    """.split()),
    test_command_tuple=("ctloom-test-r",),
    test_path="tests/testthat", test_target="path",
    run_command_tuple=("Rscript",),
    test_name_tuple=("test-*.[Rr]", "test_*.[Rr]"),
    dep_file_tuple=("DESCRIPTION", "renv.lock"),
    # Tells ctloom-test-r to install no package
    offline_env_tuple=(("CTLOOM_TEST_R_OFFLINE", "1"),),
    check_command_tuple=("ctloom-check-r",),
    build_output_tuple=("testthat-problems.rds",),
    compile_error_line=")",
    value_type_set=frozenset({"binary_operator"}),
    function_value_pattern=r"\bfunction\s*\(|\\\(",
    comment_mark_tuple=("#",))

# The extensions codetwine parses, in lower case, each mapped to its LangConfig.
# codetwine parses them in any case where the language says so
LANG_DICT: dict[str, LangConfig] = {
    "py": _PY, "java": _JAVA, "kt": _KOTLIN, "kts": _KOTLIN,
    "js": _JS, "jsx": _JS, "mjs": _JS, "cjs": _JS,
    "ts": _TS, "tsx": _TS, "mts": _TS, "cts": _TS,
    "c": _C, "cpp": _CPP, "h": _CPP, "cc": _CPP, "cxx": _CPP, "hpp": _CPP, "hh": _CPP,
    "hxx": _CPP, "rs": _RS, "sql": _SQL,
    "cbl": _CBL, "cob": _CBL, "cpy": _CBL, "bms": _BMS,
    "jcl": _JCL, "prc": _JCL, "proc": _JCL, "csd": _CSD, "cs": _CS,
    "r": _R, "rmd": _R, "qmd": _R,
}


# The extensions that count as another language's when the primary language is settled,
# each with the language they count as and the language that keeps them their own. A
# repository whose headers outnumber its sources is the first language when it holds no
# file of the second; a repository of COBOL programs and their BMS screen maps, JCL and
# CSD files is a COBOL repository whatever the number of those. None is no language
HEADER_OWNER_TUPLE: tuple[tuple[str, str, str | None], ...] = (
    ("h", "c", "cpp"), ("bms", "cbl", None), ("jcl", "cbl", None), ("prc", "cbl", None),
    ("proc", "cbl", None), ("csd", "cbl", None))


def header_owner(ext: str, ext_count_dict: dict[str, int]) -> str:
    """Return the language a header extension stands for in one index.

    Examples:
        ("h", {"h": 9, "c": 4}) -> "c"
        ("h", {"h": 9, "cpp": 4}) -> "h"
        ("bms", {"bms": 9, "cpy": 4}) -> "cbl"
        ("jcl", {"jcl": 9, "cbl": 4}) -> "cbl"

    Args:
        ext: The extension to look up.
        ext_count_dict: {extension: how many files of it the index holds}.

    Returns:
        The extension whose settings are to be used, which is ext itself unless it is a
        header extension of a language the index holds, through any of its extensions,
        and of no other.
    """
    lang_set = {ALIAS_DICT.get(own, own) for own, count in ext_count_dict.items()
                if count and own != ext}
    for header, owner, other in HEADER_OWNER_TUPLE:
        if ext == header and owner in lang_set and other not in lang_set:
            return owner
    return ext


def _alias_dict() -> dict[str, str]:
    """Work out which extensions are another extension's, from LANG_DICT itself.

    Extensions sharing one LangConfig are one language: the first key holding a
    settings object is the one the others are aliases of.

    Returns:
        {alias extension: the extension it shares its settings with}, e.g.
        {"kts": "kt", "jsx": "js", "tsx": "ts", "h": "cpp"}.
    """
    first_ext_dict: dict[int, str] = {}
    alias_dict = {}
    for ext, lang_config in LANG_DICT.items():
        base = first_ext_dict.setdefault(id(lang_config), ext)
        if base != ext:
            alias_dict[ext] = base
    return alias_dict


def name_only_dir_set(output_tuple: tuple[str, ...]) -> frozenset[str]:
    """Return the directory names a build_output_tuple gives by name alone.

    Examples:
        ("node_modules/", "/dist/", "uv.lock", ".yarn/install-state.gz")
            -> {"node_modules"}

    Args:
        output_tuple: A LangConfig build_output_tuple.

    Returns:
        The entries ending in "/" with no other "/", without it, read as fnmatch
        patterns ("*.egg-info").
    """
    return frozenset(pattern[:-1] for pattern in output_tuple
                     if pattern.endswith("/") and "/" not in pattern[:-1])


def split_output_tuple(output_tuple: tuple[str, ...]
                       ) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a build_output_tuple into its entries not starting with "/" and the others.

    Args:
        output_tuple: A LangConfig build_output_tuple.

    Returns:
        (the entries not starting with "/", those starting with "/"), each in order.
    """
    return (tuple(pattern for pattern in output_tuple if not pattern.startswith("/")),
            tuple(pattern for pattern in output_tuple if pattern.startswith("/")))


def root_target_pattern_tuple(lang_config: LangConfig) -> tuple[str, ...]:
    """Return every pattern of a language's root_target_tuple, group after group.

    Args:
        lang_config: The language's settings.

    Returns:
        The fnmatch patterns, () for a language with no root target.
    """
    return tuple(pattern for group in lang_config.root_target_tuple for pattern in group)


def project_file_pattern_tuple(lang_config: LangConfig) -> tuple[str, ...]:
    """Return the patterns of the files that make a directory one of a language's projects.

    Args:
        lang_config: The language's settings.

    Returns:
        The fnmatch patterns of its dep_file_tuple, root_target_tuple,
        test_build_file_tuple, run_file_tuple and project_file_tuple, () for a language
        with none.
    """
    return (*lang_config.dep_file_tuple, *root_target_pattern_tuple(lang_config),
            *lang_config.test_build_file_tuple, *lang_config.run_file_tuple,
            *lang_config.project_file_tuple)


def is_dep_file(path: str, pattern_tuple: tuple[str, ...] | frozenset[str]) -> bool:
    """Tell whether a path is a dependency file by its base name.

    Examples:
        ("src/App/App.csproj", ("*.csproj",))    -> True
        ("backend/pyproject.toml", ("pyproject.toml",)) -> True
        ("src/App/App.cs", ("*.csproj",))        -> False

    Args:
        path: A repository-relative path.
        pattern_tuple: dep_file_tuple entries, fnmatch patterns matched with case.

    Returns:
        True when the base name matches one of the patterns.
    """
    name = path.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in pattern_tuple)


def dep_file_example(pattern: str) -> str:
    """Return a file name a dependency file pattern stands for.

    Examples:
        "*.csproj"       -> "App.csproj"
        "pyproject.toml" -> "pyproject.toml"

    Args:
        pattern: One dep_file_tuple entry.

    Returns:
        The pattern with each "*" written as "App".
    """
    return pattern.replace("*", "App")


# The dependency file names and patterns of the languages of LANG_DICT, put together
DEP_FILE_TUPLE: tuple[str, ...] = tuple(sorted(
    {name for lang_config in LANG_DICT.values() for name in lang_config.dep_file_tuple}))

# The offline settings of the build tools of the languages of LANG_DICT, put together
OFFLINE_ENV_DICT: dict[str, str] = {name: value for lang_config in LANG_DICT.values()
                                    for name, value in lang_config.offline_env_tuple}

# The extensions that share another extension's LangConfig, each with that extension
ALIAS_DICT: dict[str, str] = _alias_dict()

# The language name injected into the instructions, one per LANG_DICT key
LANG_NAME_DICT: dict[str, str] = {
    "py": "Python", "java": "Java", "kt": "Kotlin", "kts": "Kotlin",
    "js": "JavaScript", "jsx": "JavaScript", "mjs": "JavaScript", "cjs": "JavaScript",
    "ts": "TypeScript", "tsx": "TypeScript", "mts": "TypeScript", "cts": "TypeScript",
    "c": "C", "cpp": "C++", "h": "C++", "cc": "C++", "cxx": "C++", "hpp": "C++",
    "hh": "C++", "hxx": "C++", "rs": "Rust", "sql": "SQL",
    "cbl": "COBOL", "cob": "COBOL", "cpy": "COBOL", "bms": "BMS",
    "jcl": "JCL", "prc": "JCL", "proc": "JCL", "csd": "CICS CSD", "cs": "C#",
    "r": "R", "rmd": "R Markdown", "qmd": "Quarto",
}

# The example path injected into the instructions, one per LANG_DICT key
PATH_EXAMPLE_DICT: dict[str, str] = {
    "py": "src/config.py", "java": "src/main/java/com/example/Config.java",
    "kt": "src/main/kotlin/com/example/Config.kt", "kts": "build.gradle.kts",
    "js": "src/config.js", "jsx": "src/Config.jsx", "mjs": "src/config.mjs",
    "cjs": "src/config.cjs",
    "ts": "src/config.ts", "tsx": "src/Config.tsx", "mts": "src/config.mts",
    "cts": "src/config.cts",
    "c": "src/config.c", "cpp": "src/config.cpp", "h": "src/config.h",
    "cc": "src/config.cc", "cxx": "src/config.cxx", "hpp": "src/config.hpp",
    "hh": "src/config.hh", "hxx": "src/config.hxx",
    "rs": "src/config.rs", "sql": "db/config.sql",
    "cbl": "src/CUSTUPD.cbl", "cob": "src/custupd.cob", "cpy": "copy/CUSTREC.cpy",
    "bms": "bms/CUSTMAP.bms", "jcl": "jcl/NIGHTLY.jcl", "prc": "proc/REPROC.prc",
    "proc": "proc/REPROC.proc", "csd": "csd/APP.csd", "cs": "src/App/Config.cs",
    "r": "R/config.R", "rmd": "vignettes/config.Rmd", "qmd": "docs/config.qmd",
}

# The example of a definition's first line injected into the instructions, one per
# LANG_DICT key
SIGNATURE_EXAMPLE_DICT: dict[str, str] = {
    "py": "def load_config(key, default=None):",
    "java": "public Config loadConfig(String key)",
    "kt": "fun loadConfig(key: String): Config",
    "kts": "fun loadConfig(key: String): Config",
    "js": "function loadConfig(key, fallback) {",
    "jsx": "function loadConfig(key, fallback) {",
    "mjs": "export function loadConfig(key, fallback) {",
    "cjs": "function loadConfig(key, fallback) {",
    "ts": "function loadConfig(key: string): Config {",
    "tsx": "function loadConfig(key: string): Config {",
    "mts": "export function loadConfig(key: string): Config {",
    "cts": "function loadConfig(key: string): Config {",
    "c": "Config *load_config(const char *key)",
    "cpp": "Config load_config(const std::string &key)",
    "h": "Config load_config(const std::string &key);",
    "cc": "Config load_config(const std::string &key)",
    "cxx": "Config load_config(const std::string &key)",
    "hpp": "Config load_config(const std::string &key);",
    "hh": "Config load_config(const std::string &key);",
    "hxx": "Config load_config(const std::string &key);",
    "rs": "pub fn load_config(key: &str) -> Config {",
    "sql": "CREATE FUNCTION load_config(key TEXT) RETURNS TEXT AS $$",
    "cbl": "       1000-LOAD-CONFIG.",
    "cob": "       1000-LOAD-CONFIG.",
    "cpy": "       01  CONFIG-RECORD.",
    "bms": "CFGMAP   DFHMDI SIZE=(24,80)",
    "jcl": "//NIGHTLY JOB (ACCT),CLASS=A,MSGCLASS=X",
    "prc": "//REPROC  PROC HLQ=PROD",
    "proc": "//REPROC  PROC HLQ=PROD",
    "csd": "Define Transaction(TX01) Group(APPGRP)",
    "cs": "public Config LoadConfig(string key)",
    "r": "load_config <- function(key, default = NULL) {",
    "rmd": "load_config <- function(key, default = NULL) {",
    "qmd": "load_config <- function(key, default = NULL) {",
}

# What starting this project tends to look like, one per LANG_DICT key: the example
# given when RUN_COMMAND is not set and the language has no default launcher
RUN_EXAMPLE_DICT: dict[str, str] = {
    "py": "python -m yourpackage", "java": "java -jar target/app.jar",
    "kt": "gradle run", "kts": "gradle run",
    "js": "node index.js", "jsx": "node index.js", "mjs": "node index.mjs",
    "cjs": "node index.cjs",
    "ts": "npm start", "tsx": "npm start", "mts": "npm start", "cts": "npm start",
    "c": "./build/app", "cpp": "./build/app", "h": "./build/app", "cc": "./build/app",
    "cxx": "./build/app", "hpp": "./build/app", "hh": "./build/app", "hxx": "./build/app",
    "rs": "cargo run --", "sql": "psql -f db/config.sql",
    "cbl": "cobc -x -j -I copy src/MAIN.cbl", "cob": "cobc -x -j -I copy src/main.cob",
    "cpy": "cobc -x -j -I copy src/MAIN.cbl", "bms": "cobc -x -j -I copy src/MAIN.cbl",
    "jcl": "zowe zos-jobs submit local-file jcl/NIGHTLY.jcl",
    "prc": "zowe zos-jobs submit local-file jcl/NIGHTLY.jcl",
    "proc": "zowe zos-jobs submit local-file jcl/NIGHTLY.jcl",
    "csd": "zowe zos-jobs submit local-file jcl/CSDUPD.jcl",
    "cs": "dotnet run --project src/App",
    "r": "Rscript R/main.R", "rmd": "Rscript -e 'rmarkdown::render(\"report.Rmd\")'",
    "qmd": "quarto render report.qmd",
}

# The definition name the instructions use in their examples, one per LANG_DICT key
DEF_EXAMPLE_DICT: dict[str, str] = {
    "py": "load_config", "java": "loadConfig", "kt": "loadConfig", "kts": "loadConfig",
    "js": "loadConfig", "jsx": "loadConfig", "mjs": "loadConfig", "cjs": "loadConfig",
    "ts": "loadConfig", "tsx": "loadConfig", "mts": "loadConfig", "cts": "loadConfig",
    "c": "load_config", "cpp": "load_config", "h": "load_config", "cc": "load_config",
    "cxx": "load_config", "hpp": "load_config", "hh": "load_config",
    "hxx": "load_config", "rs": "load_config",
    "sql": "load_config", "cbl": "1000-LOAD-CONFIG", "cob": "1000-LOAD-CONFIG",
    "cpy": "CONFIG-RECORD", "bms": "CFGMAPI", "jcl": "NIGHTLY", "prc": "REPROC",
    "proc": "REPROC", "csd": "TX01", "cs": "LoadConfig",
    "r": "load_config", "rmd": "load_config", "qmd": "load_config",
}


def split_def_key(def_key: str) -> tuple[str, str]:
    """Split a definition key into its file path and the definition's name.

    The key is "<file path>::<definition name>", with "#<type>#<start line>" behind the
    name where a file holds several definitions of it. The split is at the first "::":
    a name can hold "::" itself (C++: a member defined outside its class).

    Examples:
        "p/config_py/config.py::load_config" -> ("p/config_py/config.py", "load_config")
        "p/shape_cpp/shape.cpp::Shape::count" -> ("p/shape_cpp/shape.cpp", "Shape::count")
        "p/config_py/config.py"              -> ("p/config_py/config.py", "")

    Args:
        def_key: A definition key from the index, with a knowledge path or a
            repository-relative path, or a file path alone (a key of the file index).

    Returns:
        (file path, name with its suffix); the name is "" for a file path alone.
    """
    path, _, name = def_key.partition("::")
    return path, name


# The "#<type>#<start line>" def_key_suffix() writes, at the end of a key or a name
_DEF_KEY_SUFFIX_RE = re.compile(r"#[^#]+#\d+$")


def def_key_suffix(def_type: str, start_line: int) -> str:
    """Return what a definition's key ends with where its file holds its name twice.

    Examples:
        ("constructor_declaration", 5) -> "#constructor_declaration#5"

    Args:
        def_type: The definition's type, as the analysis gives it.
        start_line: The line the definition starts on.

    Returns:
        "#<type>#<start line>".
    """
    return f"#{def_type}#{start_line}"


def name_without_suffix(name: str) -> str:
    """Return a definition's name without the suffix def_key_suffix() writes.

    Examples:
        "Box#constructor_declaration#5" -> "Box"
        "#count"                        -> "#count"
        "#count#field_definition#9"     -> "#count"

    Args:
        name: A definition's name, or a key, with or without the suffix.

    Returns:
        The argument without the "#<type>#<start line>" at its end. A "#" anywhere
        else stays.
    """
    return _DEF_KEY_SUFFIX_RE.sub("", name)


def ext_of(path: str) -> str:
    """Return a path's extension in lower case, or "" when it has none.

    Examples:
        "myapp/config/settings.py" -> "py"
        "docs/Makefile"            -> ""

    Args:
        path: A path including its extension.

    Returns:
        The lower-case extension without the leading dot, or "" when there is no dot.
    """
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


# {path: LANG_DICT key}: the files the analysis read in a language their extension does
# not say, under their knowledge path and their repository-relative path
_FILE_LANG_DICT: dict[str, str] = {}


def file_lang(file_entry: dict) -> str:
    """Return the language the analysis read a file in, where its extension does not say it.

    Examples:
        {"file": "p/dcl/DCLX_dcl/DCLX.dcl", "file_dependencies": {"language": "cpy"}}
            -> "cpy"
        {"file": "p/src/A_cbl/A.Cbl", "file_dependencies": {"language": "cbl"}} -> ""

    Args:
        file_entry: One file entry of the analysis. An entry without "language" in its
            file_dependencies is read by its extension.

    Returns:
        The extension whose language the analysis read the file in, or "" when that is
        ext_of() of the file or the file has no language.
    """
    lang = file_entry.get("file_dependencies", {}).get("language", "")
    return lang if lang != ext_of(file_entry["file"]) else ""


def add_file_lang(lang_dict: dict[str, str]) -> None:
    """Add files read in a language their extension does not say to _FILE_LANG_DICT.

    Args:
        lang_dict: {knowledge path: file_lang() of it} of the files where file_lang()
            is not "".
    """
    for knowledge_path, lang in lang_dict.items():
        _FILE_LANG_DICT[knowledge_path] = lang
        _FILE_LANG_DICT[copy_path_to_rel(knowledge_path.partition("/")[2])] = lang


def set_file_lang(lang_dict: dict[str, str]) -> None:
    """Replace _FILE_LANG_DICT with files read in a language their extension does not say.

    Args:
        lang_dict: {knowledge path: file_lang() of it} of the files of one index where
            file_lang() is not "".
    """
    _FILE_LANG_DICT.clear()
    add_file_lang(lang_dict)


def lang_ext(path: str) -> str:
    """Return the LANG_DICT key a file is read by.

    Examples:
        "src/MAIN.Cbl"  -> "cbl"
        "dcl/DCLX.dcl"  -> "cpy" (when the analysis read it as a COBOL copybook)
        "docs/Makefile" -> ""

    Args:
        path: A knowledge path or a repository-relative path; the path part of a
            definition key works too.

    Returns:
        The language _FILE_LANG_DICT holds for the file, otherwise ext_of(path).
    """
    return _FILE_LANG_DICT.get(path) or ext_of(path)


def lang_of(path: str) -> LangConfig | None:
    """Look a file's settings up. None for a file of no language covered.

    Examples:
        "myapp/config/settings.py" -> _PY
        "docs/Makefile"            -> None

    Args:
        path: A path including its extension; the path part of a definition key works
            too.

    Returns:
        The LangConfig of lang_ext(path), or None when it is not in LANG_DICT.
    """
    return LANG_DICT.get(lang_ext(path))


def edit_note(ext: str, tool_name_set: Set[str]) -> str:
    """Return what a stage that writes code is told about a language.

    Args:
        ext: A LANG_DICT key, the primary language.
        tool_name_set: The names of the tools the stage is handed.

    Returns:
        The pieces of the language's edit_note_tuple that name no tool, and those that
        name a tool the stage is handed, one after another. "" when there is none.
    """
    return "".join(text for tool_name, text in LANG_DICT[ext].edit_note_tuple
                   if not tool_name or tool_name in tool_name_set)


def edit_paragraph(ext: str, tool_name_set: Set[str]) -> str:
    """Return a language's edit_note() as a paragraph that follows another one.

    Args:
        ext: A LANG_DICT key, the primary language.
        tool_name_set: The names of the tools the stage is handed.

    Returns:
        A blank line, the note and a newline, or "" when the language has no note.
    """
    note = edit_note(ext, tool_name_set).rstrip("\n")
    return f"\n{note}\n" if note else ""


def is_value_type(path: str, def_type: str, text: str) -> bool:
    """Say whether a definition gives a variable, a constant or a field a plain value.

    A plain value holds no function and no class. The definition is read by the rules of
    the file's language.

    Examples:
        ("web/util.js", "lexical_declaration", "const LIMIT = 3;")       -> True
        ("web/util.js", "lexical_declaration", "const twice = (n) => n") -> False
        ("web/util.js", "function_declaration", "function f() {}")       -> False

    Args:
        path: The file the definition is in.
        def_type: The definition's type, as the analysis names it.
        text: The definition's own text, as the analysis holds it.

    Returns:
        True when the type is in the value_type_set of the file's language and
        neither the text nor the text without its comments (_no_comment_text) holds
        anything its function_value_pattern matches.
    """
    lang_config = lang_of(path)
    if lang_config is None or def_type not in lang_config.value_type_set:
        return False
    pattern = lang_config.function_value_pattern
    if not pattern:
        return True
    if re.search(pattern, text):
        return False
    bare_text = _no_comment_text(text, lang_config.comment_mark_tuple)
    return bare_text == text or not re.search(pattern, bare_text)


def _no_comment_text(text: str, mark_tuple: tuple[str, ...]) -> str:
    """Return a text with a space in place of each of its comments.

    A comment is read by its marks alone, wherever they stand: a mark inside a
    string starts a comment as one outside it does.

    Examples:
        ("a = [] /* x */ (b)", ("//", "/*", "*/")) -> "a = []   (b)"
        ("run() // one\\n{ }", ("//", "/*", "*/")) -> "run()  \\n{ }"

    Args:
        text: The text.
        mark_tuple: A language's comment_mark_tuple.

    Returns:
        The text, as it is when it holds no comment or mark_tuple is empty. A
        comment that runs over lines and is not ended runs to the end of the text.
    """
    if not mark_tuple:
        return text
    line_mark, block_start, block_end = (*mark_tuple, "", "")[:3]
    start_re = re.compile("|".join(re.escape(mark) for mark in (block_start, line_mark)
                                   if mark))
    part_list, position = [], 0
    while (match := start_re.search(text, position)) is not None:
        part_list.append(text[position:match.start()])
        if match.group() == line_mark:
            end = text.find("\n", match.end())
        else:
            end = text.find(block_end, match.end())
            end = end + len(block_end) if end != -1 else -1
        if end == -1:
            position = len(text)
            break
        position = end
    if not part_list:
        return text
    return " ".join([*part_list, text[position:]])


def own_name(path: str, name: str) -> str:
    """Return a definition's name without the names of the definitions it belongs to.

    A name made out of its type's name, such as a destructor's, is given as the type's
    name.

    Examples:
        ("src/calc.cpp", "Calc::mul")               -> "mul"
        ("src/calc.cpp", "ns::Calc::operator bool") -> "operator bool"
        ("src/calc.cpp", "Calc::~Calc")             -> "Calc"
        ("R/stats.R", "print.myclass")              -> "print.myclass"

    Args:
        path: The file the name belongs to.
        name: The name, as the analysis gives it.

    Returns:
        The name with what the name_prefix_pattern of the file's language matches
        taken off; the name unchanged where the language has no such pattern or
        nothing would be left.
    """
    lang_config = lang_of(path)
    if lang_config is None or not lang_config.name_prefix_pattern:
        return name
    return re.sub(lang_config.name_prefix_pattern, "", name, count=1) or name


def name_fold(path: str) -> Callable[[str], str]:
    """Return what the definition names of a file are compared through.

    Args:
        path: The file the names belong to; a definition key's path part works too.

    Returns:
        str.upper when the file's language ignores case, otherwise str.
    """
    lang_config = lang_of(path)
    return str.upper if lang_config is not None and lang_config.ignore_case else str


def name_key(path: str, name: str) -> str:
    """Return the form a definition name of a file is compared in.

    Examples:
        ("src/MAIN.cbl", "Main-Para") -> "MAIN-PARA"
        ("app/main.py", "Main")       -> "Main"

    Args:
        path: The file the name belongs to; a definition key's path part works too.
        name: The name.

    Returns:
        name_fold(path)(name): the name in upper case when the file's language ignores
        case, otherwise the name unchanged.
    """
    return name_fold(path)(name)
