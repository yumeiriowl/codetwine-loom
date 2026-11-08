"""Index steps 1-2."""

import pytest

from codetwine_loom.index.build import build_index
from codetwine_loom.index.graph import (add_def_node, build_def_node_dict,
                                           build_call_list, key_by_file_name_dict,
                                           outer_def, def_by_file_dict,
                                           no_resolve_ref_dict, resolve_usage,
                                           signature_index, signature_of,
                                           target_key_dict)
from codetwine_loom.index import store
from codetwine_loom.index.store import is_constant
from codetwine_loom.languages import (LANG_DICT, def_key_suffix, name_without_suffix,
                                       split_def_key)


def _calls(knowledge, defs):
    """Build the calls edges over an analysis, as build_index does."""
    usage_dict = {f["file"]: f["file_dependencies"]["callee_usages"]
                  for f in knowledge.iter_file()}
    return build_call_list(usage_dict, key_by_file_name_dict(defs),
                           def_by_file_dict(defs))


def test_same_named_definitions_are_split_by_type_and_start_line(
        java_pk, knowledge_of):
    """A class and 2 constructors of its name: bare keys alone turn 5 into 3."""
    defs = build_def_node_dict(knowledge_of(java_pk))
    assert len(defs) == 5
    file = java_pk["files"][0]["file"]
    # whichever came first (the enclosing class) keeps the bare key
    assert defs[f"{file}::Config"]["type"] == "class_declaration"
    assert f"{file}::Config#constructor_declaration#5" in defs
    assert f"{file}::Config#constructor_declaration#9" in defs


def test_the_type_alone_is_not_enough(java_pk, knowledge_of):
    """Config() and Config(int) are both a constructor_declaration named Config."""
    defs = build_def_node_dict(knowledge_of(java_pk))
    ctors = [k for k, d in defs.items()
             if d["type"] == "constructor_declaration"]
    assert len(ctors) == 2


def test_a_definition_with_a_reversed_line_range_is_skipped(java_pk, knowledge_of,
                                                           caplog):
    java_pk["files"][0]["file_dependencies"]["definitions"].append(
        {"name": "broken", "type": "method_declaration",
         "start_line": 30, "end_line": 20, "context": "void broken() {"})
    defs = build_def_node_dict(knowledge_of(java_pk))
    assert not any(d["name"] == "broken" for d in defs.values())


def test_the_signature_is_the_line_that_declares_the_definition(java_pk, knowledge_of):
    defs = build_def_node_dict(knowledge_of(java_pk))
    file = java_pk["files"][0]["file"]
    assert defs[f"{file}::getApiBase"]["signature"] == \
        "public static String getApiBase() {"


@pytest.mark.parametrize("context, name, expected", [
    ("@property\ndef size(self):\n    return 1", "size", "def size(self):"),
    ("@app.route(\"/x\")\ndef page():", "page", "def page():"),
    ("@Override\npublic String toString() {", "toString",
     "public String toString() {"),
    ("def plain(x):\n    pass", "plain", "def plain(x):"),
    ("// nothing names it\n{", "gone", "// nothing names it"),
    ("@property\ndef prop(self):\n    return 2", "prop", "def prop(self):"),
    ("@app.route('/login')\ndef login():", "login", "def login():"),
    ('@GetMapping("/users")\npublic String users() {', "users",
     "public String users() {"),
    ('[HttpPost("Login")]\npublic string Login(string u) {', "Login",
     "public string Login(string u) {"),
    ("#[route(main)]\nfn main() {", "main", "fn main() {"),
    ('@app.route(\n    "/login",\n    methods=["POST"])\ndef login():', "login",
     "def login():"),
    ("[[nodiscard]] int size();", "size", "[[nodiscard]] int size();"),
    ("int sizes;\nint size;", "size", "int size;"),
    ("x = foo_bar\nfoo", "foo", "foo"),
    ("class Foo_Bar:", "Foo", "class Foo_Bar:"),
])
def test_a_decorator_or_an_annotation_is_not_the_signature(context, name, expected):
    """The line range of a decorated definition starts at the decorator."""
    assert signature_of(context, name) == expected


def test_a_name_joined_with_hyphens_is_a_whole_word_in_such_a_language():
    context = "05 CUST-ACCT-ID PIC X.\n05 CUST-ACCT PIC X."
    assert signature_index(context, "CUST-ACCT") == 0
    assert signature_index(context, "CUST-ACCT", hyphen_name=True) == 1


def test_the_signature_of_a_file_whose_language_joins_names_with_hyphens():
    """A COBOL entry without "name_line" is read with "-" as part of a word."""
    def_dict: dict = {}
    add_def_node(def_dict, {"file": "p/a_cbl/a.cbl", "file_dependencies": {
        "definitions": [{"name": "CUST-ACCT", "type": "data_description",
                         "start_line": 3, "end_line": 4,
                         "context": "05 CUST-ACCT-ID PIC X.\n05 CUST-ACCT PIC X."}]}})
    assert def_dict["p/a_cbl/a.cbl::CUST-ACCT"]["signature_line"] == 4


def test_a_use_of_the_files_own_definition_is_a_calls_edge(knowledge_of):
    """helper is used on line 6, inside caller, and LIMIT on line 3, inside helper."""
    file = "p/one_py/one.py"
    pk = {"files": [{"file": file, "file_dependencies": {
        "definitions": [
            {"name": "LIMIT", "type": "expression_statement", "start_line": 1,
             "end_line": 1, "context": "LIMIT = 3"},
            {"name": "helper", "type": "function_definition", "start_line": 2,
             "end_line": 3, "context": "def helper(x):\n    return x + LIMIT"},
            {"name": "caller", "type": "function_definition", "start_line": 5,
             "end_line": 6, "context": "def caller(x):\n    return helper(x)"}],
        "callee_usages": [],
        "same_file_usages": [{"name": "LIMIT", "lines": [3], "target_name": "LIMIT",
                              "target_start_line": 1},
                             {"name": "helper", "lines": [6], "target_name": "helper",
                              "target_start_line": 2}],
        "caller_usages": []}}]}
    assert build_index(knowledge_of(pk), "p")["edges"]["calls"] == [
        (f"{file}::caller", f"{file}::helper"), (f"{file}::helper", f"{file}::LIMIT")]


def test_an_analysis_without_same_file_usages_builds_as_before(java_pk, knowledge_of):
    assert "same_file_usages" not in java_pk["files"][0]["file_dependencies"]
    assert build_index(knowledge_of(java_pk), "app")["edges"]["calls"] == []


def test_build_file_nodes_holds_the_summary_and_the_definition_list(
        java_pk, knowledge_of):
    files = build_index(knowledge_of(java_pk), "app")["nodes"]["files"]
    file = java_pk["files"][0]["file"]
    assert files[file]["summary"].startswith("**Config.java**")
    assert len(files[file]["defs"]) == 5


def test_build_file_nodes_leaves_the_summary_empty_without_a_design_document(
        java_pk, knowledge_of):
    """A build with docs switched off writes no "doc" key; the definitions still
    come."""
    del java_pk["files"][0]["doc"]
    files = build_index(knowledge_of(java_pk), "app")["nodes"]["files"]
    file = java_pk["files"][0]["file"]
    assert files[file]["summary"] == ""
    assert len(files[file]["defs"]) == 5


def test_enclosing_def_returns_the_narrowest_definition(java_pk, knowledge_of):
    """Line 6 falls inside both the class (3-20) and the constructor (5-7)."""
    defs = build_def_node_dict(knowledge_of(java_pk))
    by_file = def_by_file_dict(defs)
    file = java_pk["files"][0]["file"]
    assert outer_def(by_file, file, 6) == \
        f"{file}::Config#constructor_declaration#5"


def test_enclosing_def_is_none_when_nothing_encloses_the_line(java_pk, knowledge_of):
    defs = build_def_node_dict(knowledge_of(java_pk))
    by_file = def_by_file_dict(defs)
    assert outer_def(by_file, java_pk["files"][0]["file"], 999) is None


def _target_map(path: str, *entry_list: tuple[str, int, int]) -> dict:
    """Build target_key_dict() for one file.

    Each entry is (definition name, start line, end line); the keys are named as
    add_def_node() names them.
    """
    defs: dict = {}
    for name, start, end in entry_list:
        key = f"{path}::{name}"
        if key in defs:
            key = f"{key}#x#{start}"
        defs[key] = {"name": name, "file": path, "lines": [start, end]}
    return target_key_dict(key_by_file_name_dict(defs), def_by_file_dict(defs))


def _usage(path: str, name: str, target_name: str | None, start_line: int | None) -> dict:
    return {"name": name, "from": path, "lines": [1], "target_name": target_name,
            "target_start_line": start_line}


def test_a_usage_is_resolved_by_the_definition_the_analysis_names():
    """The name a file writes need not be the name of the definition it leads to."""
    path = "p/m_js/m.js"
    target_dict = _target_map(path, ("helper", 1, 3), ("Shape::count", 5, 8),
                                        ("Kind", 10, 14))
    for name, target_name, line, key in (
            ("hp", "helper", 1, "helper"),                  # imported under another name
            ("Shape::count", "Shape::count", 5, "Shape::count"),
            ("Round", "Kind", 10, "Kind")):                 # a variant leads to its enum
        assert resolve_usage(target_dict, _usage(path, name, target_name, line)) \
            == [f"{path}::{key}"]


def test_a_usage_that_leads_to_no_definition_resolves_to_nothing():
    """A module used as a value: its name may be a definition of the file all the same."""
    path = "p/m_js/m.js"
    target_dict = _target_map(path, ("m", 1, 1))
    assert resolve_usage(target_dict, _usage(path, "m", None, None)) == []


@pytest.mark.parametrize("member", ["callee_usages", "same_file_usages"])
def test_an_analysis_whose_usages_name_no_definition_is_refused(member, knowledge_of):
    """An analysis an older codetwine made: no edge could be built from it."""
    pk = _two_file_pk()
    usage = _helper_usage([2])
    del usage["target_name"], usage["target_start_line"]
    pk["files"][0]["file_dependencies"][member] = [usage]
    with pytest.raises(RuntimeError, match="out of date"):
        list(knowledge_of(pk).iter_file())


def test_overloads_and_the_constructors_of_a_class_are_one_target():
    """Which of them a reference means cannot be told from the reference."""
    path = "p/C_java/C.java"
    target_dict = _target_map(
        path, ("Config", 1, 30), ("Config", 3, 5), ("Config", 7, 9),
        ("load", 11, 13), ("load", 15, 17))
    class_key_list = [f"{path}::Config", f"{path}::Config#x#3", f"{path}::Config#x#7"]
    assert resolve_usage(target_dict, _usage(path, "Config", "Config", 1)) \
        == class_key_list
    assert resolve_usage(target_dict, _usage(path, "Config", "Config", 7)) \
        == class_key_list
    assert resolve_usage(target_dict, _usage(path, "c.load", "load", 15)) \
        == [f"{path}::load", f"{path}::load#x#15"]


def test_the_same_name_in_two_classes_is_two_targets():
    path = "p/a_py/a.py"
    target_dict = _target_map(
        path, ("Engine", 1, 5), ("run", 2, 4), ("Worker", 7, 11), ("run", 8, 10))
    assert resolve_usage(target_dict, _usage(path, "Worker.run", "run", 8)) \
        == [f"{path}::run#x#8"]
    assert resolve_usage(target_dict, _usage(path, "Engine.run", "run", 2)) \
        == [f"{path}::run"]


def test_definitions_of_one_line_range_are_beside_each_other():
    """int a, b; gives two definitions over the same lines; neither holds the other."""
    path = "p/c_c/c.c"
    target_dict = _target_map(path, ("a", 3, 3), ("b", 3, 3))
    assert resolve_usage(target_dict, _usage(path, "a", "a", 3)) == [f"{path}::a"]


def _helper_usage(line_list: list[int]) -> dict:
    """A callee_usages entry of a.py for b.py's helper, as the analysis writes it."""
    return {"lines": line_list, "name": "helper", "from": "p/b_py/b.py",
            "target_name": "helper", "target_start_line": 1}


def _two_file_pk() -> dict:
    """Synthetic data where a.py uses b.py's helper on 3 lines.

    The analysis gathers one entry per (referred file, name) with every line it is
    used on, which is the shape the calls edges are built from.
    """
    return {
        "files": [
            {"file": "p/a_py/a.py",
             "file_dependencies": {
                 "definitions": [
                     {"name": "caller", "type": "function_definition",
                      "start_line": 1, "end_line": 10, "context": "def caller():"}],
                 "callee_usages": [
                     _helper_usage([2, 3, 4]),
                 ],
                 "caller_usages": []},
             "doc": {"sections": [], "summary": "a"}},
            {"file": "p/b_py/b.py",
             "file_dependencies": {
                 "definitions": [
                     {"name": "helper", "type": "function_definition",
                      "start_line": 1, "end_line": 3, "context": "def helper():"}],
                 "callee_usages": [], "caller_usages": []},
             "doc": {"sections": [], "summary": "b"}},
        ]}


def test_in_degrees_are_counted_after_calls_is_settled(knowledge_of):
    """The same (a, b) on 3 lines is still one call edge and an in-degree of 1."""
    knowledge = knowledge_of(_two_file_pk())
    calls = _calls(knowledge, build_def_node_dict(knowledge))
    assert len(calls) == 1


def test_every_line_of_one_reference_makes_its_own_edge(knowledge_of):
    """A name used from two definitions of the file is an edge from each of them."""
    pk = _two_file_pk()
    pk["files"][0]["file_dependencies"]["definitions"].append(
        {"name": "second", "type": "function_definition", "start_line": 11,
         "end_line": 20, "context": "def second():"})
    pk["files"][0]["file_dependencies"]["callee_usages"] = [
        _helper_usage([2, 12])]
    knowledge = knowledge_of(pk)
    calls = _calls(knowledge, build_def_node_dict(knowledge))
    assert calls == [("p/a_py/a.py::caller", "p/b_py/b.py::helper"),
                     ("p/a_py/a.py::second", "p/b_py/b.py::helper")]


def test_a_reference_whose_first_line_is_outside_every_definition_still_connects(
        knowledge_of):
    """A name used at the top of the file and inside a definition is that one edge."""
    pk = _two_file_pk()
    pk["files"][0]["file_dependencies"]["callee_usages"] = [
        _helper_usage([999, 2])]
    knowledge = knowledge_of(pk)
    calls = _calls(knowledge, build_def_node_dict(knowledge))
    assert calls == [("p/a_py/a.py::caller", "p/b_py/b.py::helper")]


def test_an_edge_is_drawn_to_every_definition_of_the_name(knowledge_of):
    """Which overload is meant cannot be told, so the caller is on both."""
    pk = _two_file_pk()
    pk["files"][1]["file_dependencies"]["definitions"].append(
        {"name": "helper", "type": "function_definition", "start_line": 5,
         "end_line": 7, "context": "def helper(x):"})
    knowledge = knowledge_of(pk)
    calls = _calls(knowledge, build_def_node_dict(knowledge))
    assert calls == [
        ("p/a_py/a.py::caller", "p/b_py/b.py::helper"),
        ("p/a_py/a.py::caller", "p/b_py/b.py::helper#function_definition#5")]


def test_a_reference_with_no_enclosing_definition_becomes_no_edge(knowledge_of):
    pk = _two_file_pk()
    pk["files"][0]["file_dependencies"]["callee_usages"] = [
        _helper_usage([999])]
    knowledge = knowledge_of(pk)
    calls = _calls(knowledge, build_def_node_dict(knowledge))
    assert calls == []


def test_both_ends_of_every_call_edge_are_definition_nodes(knowledge_of):
    knowledge = knowledge_of(_two_file_pk())
    defs = build_def_node_dict(knowledge)
    for a, b in _calls(knowledge, defs):
        assert a in defs and b in defs


def _renamed_pk(ext: str) -> dict:
    """Synthetic data for one language where a's call is left pointing at an old name.

    b defines new_name; a still calls old_name on 2 lines and resolve_helper on one the
    file does define. Both files are named for ext, so the same shape is built for every
    language the analysis covers.
    """
    a, b = f"p/a_{ext}/a.{ext}", f"p/b_{ext}/b.{ext}"
    return {
        "files": [
            {"file": a,
             "file_dependencies": {
                 "definitions": [
                     {"name": "caller", "type": "function", "start_line": 1,
                      "end_line": 10, "context": "caller"}],
                 "callee_usages": [
                     {"lines": [2, 4], "name": "old_name", "from": b,
                      "target_name": None, "target_start_line": None},
                     {"lines": [6], "name": "kept", "from": b,
                      "target_name": "kept", "target_start_line": 5},
                 ],
                 "caller_usages": []},
             "doc": {"sections": [], "summary": "a"}},
            {"file": b,
             "file_dependencies": {
                 "definitions": [
                     {"name": "new_name", "type": "function", "start_line": 1,
                      "end_line": 3, "context": "new_name"},
                     {"name": "kept", "type": "function", "start_line": 5,
                      "end_line": 7, "context": "kept"}],
                 "callee_usages": [], "caller_usages": []},
             "doc": {"sections": [], "summary": "b"}},
        ]}


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_a_reference_to_a_name_the_file_no_longer_defines_is_returned(ext, knowledge_of):
    """Every language the analysis covers goes through the one resolve_usage()."""
    knowledge = knowledge_of(_renamed_pk(ext))
    defs = build_def_node_dict(knowledge)
    refs = no_resolve_ref_dict(knowledge, defs)
    assert refs == {(f"p/a_{ext}/a.{ext}", f"p/b_{ext}/b.{ext}", "old_name"): [2, 4]}


def test_a_reference_that_resolves_is_not_returned(knowledge_of):
    knowledge = knowledge_of(_two_file_pk())
    assert no_resolve_ref_dict(knowledge, build_def_node_dict(knowledge)) == {}


def test_the_lines_of_one_reference_are_gathered_without_repeats(knowledge_of):
    pk = _renamed_pk("py")
    pk["files"][0]["file_dependencies"]["callee_usages"].append(
        {"lines": [2], "name": "old_name", "from": "p/b_py/b.py",
         "target_name": None, "target_start_line": None})
    knowledge = knowledge_of(pk)
    refs = no_resolve_ref_dict(knowledge, build_def_node_dict(knowledge))
    assert refs[("p/a_py/a.py", "p/b_py/b.py", "old_name")] == [2, 4]


def test_a_reference_no_definition_encloses_is_still_returned(knowledge_of):
    """build_call_list() drops it for having no node to connect; this does not."""
    pk = _renamed_pk("py")
    pk["files"][0]["file_dependencies"]["callee_usages"] = [
        {"lines": [999], "name": "old_name", "from": "p/b_py/b.py",
         "target_name": None, "target_start_line": None}]
    knowledge = knowledge_of(pk)
    refs = no_resolve_ref_dict(knowledge, build_def_node_dict(knowledge))
    assert refs == {("p/a_py/a.py", "p/b_py/b.py", "old_name"): [999]}


def test_a_reference_with_no_lines_is_returned_with_none(knowledge_of):
    pk = _renamed_pk("py")
    pk["files"][0]["file_dependencies"]["callee_usages"] = [
        {"lines": [], "name": "old_name", "from": "p/b_py/b.py",
         "target_name": None, "target_start_line": None}]
    knowledge = knowledge_of(pk)
    refs = no_resolve_ref_dict(knowledge, build_def_node_dict(knowledge))
    assert refs == {("p/a_py/a.py", "p/b_py/b.py", "old_name"): []}


def test_a_reference_to_a_name_no_file_defines_becomes_no_edge(knowledge_of, caplog):
    import logging
    pk = _two_file_pk()
    pk["files"][0]["file_dependencies"]["callee_usages"] = [
        {"lines": [2], "name": "nosuch", "from": "p/b_py/b.py",
         "target_name": None, "target_start_line": None}]
    knowledge = knowledge_of(pk)
    defs = build_def_node_dict(knowledge)
    with caplog.at_level(logging.DEBUG, logger="codetwine_loom.index.graph"):
        calls = _calls(knowledge, defs)
    assert calls == []
    assert "cannot resolve the target: nosuch" in caplog.text


def test_a_definition_key_is_split_at_the_first_separator():
    """A name can hold "::" itself (a C++ member defined outside its class)."""
    assert split_def_key("p/shape_cpp/shape.cpp::Shape::count") == \
        ("p/shape_cpp/shape.cpp", "Shape::count")
    assert split_def_key("p/C_java/C.java::Config#constructor_declaration#5") == \
        ("p/C_java/C.java", "Config#constructor_declaration#5")
    # A key of the file index is a path alone
    assert split_def_key("p/a_py/a.py") == ("p/a_py/a.py", "")


@pytest.mark.parametrize("name", ["Config", "#count", "Shape::count", "MAIN-PARA", "a.b"])
def test_the_suffix_of_a_key_is_taken_off_the_way_it_is_put_on(name):
    """A "#" a name holds itself (a private member of a JavaScript class) stays."""
    suffix = def_key_suffix("constructor_declaration", 5)
    assert suffix == "#constructor_declaration#5"
    assert name_without_suffix(name + suffix) == name_without_suffix(name) == name
    assert name_without_suffix(f"p/a_js/a.js::{name}{suffix}") == f"p/a_js/a.js::{name}"


def test_a_cpp_member_defined_outside_its_class_keeps_its_language_and_its_path():
    key = "p/shape_cpp/shape.cpp::Shape::count"
    assert store._lang_ext_of_def(key) == "cpp"
    assert is_constant(key, {"type": "declaration"}) is True
    assert is_constant(key, {"type": "function_definition"}) is False
