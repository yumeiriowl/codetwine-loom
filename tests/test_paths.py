"""Path conversion and path validation."""

import os
import pathlib

import pytest

from codetwine_loom import config, paths
from codetwine_loom.paths import (knowledge_to_repo, path_text, repo_to_knowledge,
                                     resolve_in_project, under_test_path)

PROJECT = "codetwine"


# == knowledge_to_repo and repo_to_knowledge ======================================
@pytest.mark.parametrize("knowledge, expected", [
    ("codetwine/codetwine/config/settings_py/settings.py",
     "codetwine/config/settings.py"),
    ("codetwine/main_py/main.py", "main.py"),
    ("codetwine/Makefile/Makefile", "Makefile"),                 # no extension
    ("codetwine/.gitignore/.gitignore", ".gitignore"),           # a dotfile has none either
    ("codetwine/docs/README_md/README.md", "docs/README.md"),
])
def test_knowledge_to_repo_converts_a_knowledge_path(knowledge, expected):
    assert knowledge_to_repo(knowledge, PROJECT) == expected


@pytest.mark.parametrize("path", [
    "codetwine/config/settings.py",  # already repo-relative (starts with project_name)
    "main.py",                       # already repo-relative (too few components)
    "codetwine/docs/Makefile",       # no extension
])
def test_knowledge_to_repo_raises_value_error_for_a_non_knowledge_path(path):
    with pytest.raises(ValueError):
        knowledge_to_repo(path, PROJECT)


@pytest.mark.parametrize("repo, knowledge", [
    ("codetwine/config/settings.py",
     "codetwine/codetwine/config/settings_py/settings.py"),
    ("main.py", "codetwine/main_py/main.py"),
    ("Makefile", "codetwine/Makefile/Makefile"),
    ("docs/.env.example", "codetwine/docs/.env_example/.env.example"),
])
def test_repo_to_knowledge_is_the_inverse_of_knowledge_to_repo(repo, knowledge):
    assert repo_to_knowledge(repo, PROJECT) == knowledge
    assert knowledge_to_repo(knowledge, PROJECT) == repo


# == resolve_in_project ==============
@pytest.fixture
def project(tmp_path, monkeypatch):
    """Prepare a PROJECT_DIR and an OUTPUT_DIR placed outside it."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("x = 1\n")
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    return root, out


def test_resolve_in_project_returns_an_absolute_path(project):
    root, _ = project
    assert resolve_in_project("pkg/a.py") == str(root / "pkg" / "a.py")


@pytest.mark.parametrize("path", [None, 3, ["a.py"], b"a.py"])
def test_resolve_in_project_rejects_what_is_not_a_path(project, path):
    with pytest.raises(ValueError, match="a path must be a string, not "
                                         + type(path).__name__):
        resolve_in_project(path)


def test_a_path_like_object_is_read_as_the_string_it_stands_for(project):
    root, _ = project
    assert path_text("pkg/a.py") == path_text(pathlib.PurePosixPath("pkg/a.py")) == \
        "pkg/a.py"
    assert resolve_in_project(pathlib.Path("pkg/a.py")) == str(root / "pkg" / "a.py")
    with pytest.raises(ValueError, match="a path must be a string, not int"):
        path_text(3)


def test_resolve_in_project_rejects_a_path_pointing_outside(project):
    with pytest.raises(ValueError):
        resolve_in_project("../../etc/passwd")


def test_resolve_in_project_rejects_a_path_inside_output_dir(tmp_path, monkeypatch):
    """With OUTPUT_DIR inside PROJECT_DIR, the first check alone lets it through."""
    root = tmp_path / "proj"
    out = root / "out"
    out.mkdir(parents=True)
    (out / "copy.py").write_text("x = 1\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    with pytest.raises(ValueError):
        resolve_in_project("out/copy.py")


def test_resolve_in_project_rejects_going_outside_through_a_symlink(project):
    """This is what pins down realpath rather than abspath."""
    root, _ = project
    os.symlink("/etc", root / "link")
    with pytest.raises(ValueError):
        resolve_in_project("link/passwd")


# == Which paths hold the tests ==============================
def test_the_directory_itself_and_what_is_inside_it_are_both_under_it(monkeypatch):
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    assert under_test_path(["tests", "tests/test_a.py", "tests/unit/test_b.py"]) == \
        ["tests", "tests/test_a.py", "tests/unit/test_b.py"]


def test_a_path_that_only_starts_with_the_same_letters_is_not_under_it(monkeypatch):
    """"tests_helper.py" is a file beside the directory, not a file in it."""
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    assert under_test_path(["tests_helper.py", "main.py", "a/tests_old/b.py",
                            "a/my_tests/b.py"]) == []


def test_the_test_directory_under_another_directory_holds_tests_too(monkeypatch):
    """A multi-module layout keeps one test directory per module."""
    monkeypatch.setattr(config, "TEST_PATH", "src/test/java")
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    assert under_test_path(["core/src/test/java/a/FooTest.java", "src/test/java/B.java",
                            "core/src/main/java/a/Foo.java", "core/test/java/C.java"]) == \
        ["core/src/test/java/a/FooTest.java", "src/test/java/B.java"]


@pytest.mark.parametrize("ext, path_list, expected", [
    ("js", ["src/__tests__/cfg.js", "src/cfg.test.js", "src/cfg.spec.jsx", "src/cfg.js",
            "api.spec.yaml"],
     ["src/__tests__/cfg.js", "src/cfg.test.js", "src/cfg.spec.jsx"]),
    ("ts", ["src/cfg.test.ts", "src/cfg.ts"], ["src/cfg.test.ts"]),
    ("mjs", ["src/cfg.test.mjs", "src/cfg.spec.cjs", "src/cfg.mjs", "src/cfg.test.xjs"],
     ["src/cfg.test.mjs", "src/cfg.spec.cjs"]),
    ("cjs", ["src/cfg.test.cjs", "src/cfg.cjs"], ["src/cfg.test.cjs"]),
    ("mts", ["src/cfg.test.mts", "src/cfg.spec.cts", "src/cfg.mts"],
     ["src/cfg.test.mts", "src/cfg.spec.cts"]),
    ("cts", ["src/cfg.spec.cts", "src/cfg.cts"], ["src/cfg.spec.cts"]),
    ("cs", ["App.Tests/FooTests.cs", "src/App.UnitTests/X.cs", "src/App/Foo.cs"],
     ["App.Tests/FooTests.cs", "src/App.UnitTests/X.cs"]),
    ("py", ["src/test_cfg.py", "src/cfg.test.js"], []),
])
def test_a_language_s_test_names_mark_tests_wherever_they_stand(monkeypatch, ext,
                                                                 path_list, expected):
    monkeypatch.setattr(config, "TEST_PATH", "nowhere")
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    assert under_test_path(path_list) == expected


def test_the_other_test_directories_hold_tests_too(monkeypatch):
    """With TEST_PATH_ALT_TUPLE, a file under one of its directories at the repository
    root is a test whatever its name; one under such a directory deeper down is not."""
    monkeypatch.setattr(config, "TEST_PATH", "test")
    monkeypatch.setattr(config, "PRIMARY_EXT", "ts")
    path_list = ["tests/fixtures/app/src/main.ts", "test/helper.ts", "src/main.ts",
                 "pkg/tests/main.ts"]
    assert under_test_path(path_list) == ["test/helper.ts"]
    monkeypatch.setattr(config, "TEST_PATH_ALT_TUPLE", ("tests",))
    assert under_test_path(path_list) == ["tests/fixtures/app/src/main.ts", "test/helper.ts"]
    # A package of the code named like a test directory, deeper down
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_PATH_ALT_TUPLE", ("test", "testing"))
    assert under_test_path(["pkg/testing/__init__.py", "testing/test_a.py"]) == \
        ["testing/test_a.py"]


def test_the_repository_root_as_the_test_directory_names_no_directory(monkeypatch):
    """TEST_PATH "." runs every test of the repository; which files are tests is then
    told by the language's test names alone."""
    monkeypatch.setattr(config, "TEST_PATH", ".")
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    assert under_test_path(["main.js", "tests/a.js", "src/a.test.js"]) == ["src/a.test.js"]


def test_no_test_directory_setting_puts_nothing_under_it(monkeypatch):
    """A language with no such setting names no directory, so the answer is not every
    path in the repository."""
    monkeypatch.setattr(config, "TEST_PATH", "")
    assert under_test_path(["tests/test_a.py", "main.py", "/etc/passwd"]) == []


@pytest.mark.parametrize("test_path, ext, root, has_rule", [
    ("tests", "py", "tests", True), (".", "py", "", False), ("", "py", "", False),
    (".", "js", "", True), ("", "cs", "", True), ("src/test/java", "java",
                                                  "src/test/java", True)])
def test_no_path_can_be_a_test_without_a_directory_and_without_test_names(
        monkeypatch, test_path, ext, root, has_rule):
    monkeypatch.setattr(config, "TEST_PATH", test_path)
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    assert (paths.test_root(), paths.has_test_rule()) == (root, has_rule)
    if not has_rule:
        assert under_test_path(["tests/test_a.py", "test_a.py", "a_test.py"]) == []


def test_the_paths_keep_the_order_they_came_in(monkeypatch):
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    assert under_test_path(["spec/z.rb", "main.rb", "spec/a.rb"]) == \
        ["spec/z.rb", "spec/a.rb"]


def test_an_empty_list_renders_as_nothing():
    from codetwine_loom.paths import list_text
    assert list_text([]) == ""


def test_a_path_under_the_repository_s_place_on_the_starting_machine_is_read_from_its_root(
        monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setenv(paths.HOST_PROJECT_ENV, "/home/me/link\n/home/me/myapp")
    assert paths.host_path_of("/home/me/myapp/src/a.py") == "src/a.py"
    assert paths.host_path_of("/home/me/link/src/a.py") == "src/a.py"
    assert paths.host_path_of("/home/me/myapp") == "."
    assert paths.host_path_of("/home/me/myapp2/a.py") == "/home/me/myapp2/a.py"
    assert paths.host_path_of("src/a.py") == "src/a.py"
    assert paths.to_repo_relative("/home/me/myapp/src/a.py") == "src/a.py"
    assert paths.resolve_in_project("/home/me/myapp/src/a.py") == str(root / "src" / "a.py")
    monkeypatch.delenv(paths.HOST_PROJECT_ENV)
    with pytest.raises(ValueError, match="outside the repository"):
        paths.resolve_in_project("/home/me/myapp/src/a.py")
