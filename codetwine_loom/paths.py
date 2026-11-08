"""Path notation conversion and path validation.

The paths inside the analysis (knowledge paths) have a different shape from the real
paths of the target repository (repository-relative paths).

    knowledge path : <project name>/<directory>/<stem>_<ext>/<basename>
    repo-relative  : <directory>/<basename>
    absolute       : <PROJECT_DIR>/<repo-relative path>

The <stem>_<ext> directory is the one codetwine wraps each copy in; a file without an
extension is wrapped in a directory of its own name (Makefile/Makefile). Both
conversions are codetwine's rel_to_copy_path() and copy_path_to_rel().
"""

import fnmatch
import os
from collections.abc import Iterable, Set

from codetwine.utils.file_utils import copy_path_to_rel, rel_to_copy_path

from codetwine_loom import config, gitcmd, tmpfile
from codetwine_loom.languages import LANG_DICT, PATH_EXAMPLE_DICT


def knowledge_to_repo(knowledge_path: str, project_name: str) -> str:
    """Convert a knowledge path into a repository-relative path.

    A path counts as a knowledge path when it starts with project_name and the rest
    has the shape codetwine's copy_path_to_rel() recognises.

    Examples:
        "myapp/myapp/config/settings_py/settings.py"
            -> "myapp/config/settings.py"
        "myapp/main_py/main.py" -> "main.py"
        "myapp/Makefile/Makefile" -> "Makefile"
        "myapp/config/settings.py" -> ValueError (already repository-relative)

    Args:
        knowledge_path: A knowledge path.
        project_name: The directory name of PROJECT_DIR, e.g. "myapp".

    Returns:
        A repository-relative path.

    Raises:
        ValueError: When it does not have the shape of a knowledge path.
    """
    prefix, _, copy_path = knowledge_path.partition("/")
    repo_path = copy_path_to_rel(copy_path) if copy_path else ""
    # copy_path_to_rel() returns its argument unchanged when the shape does not match
    if prefix != project_name or repo_path == copy_path:
        raise ValueError(f"not a knowledge path: {knowledge_path}")
    return repo_path


def repo_to_knowledge(repo_path: str, project_name: str) -> str:
    """Convert a repository-relative path into a knowledge path.

    The inverse of knowledge_to_repo.

    Examples:
        "myapp/config/settings.py"
            -> "myapp/myapp/config/settings_py/settings.py"
        "main.py" -> "myapp/main_py/main.py"
        "Makefile" -> "myapp/Makefile/Makefile"

    Args:
        repo_path: A repository-relative path.
        project_name: The directory name of PROJECT_DIR.

    Returns:
        A knowledge path.
    """
    return f"{project_name}/{rel_to_copy_path(repo_path)}"


def to_repo_path(path: str, project_name: str) -> str:
    """Convert to a repository-relative path, but only for a knowledge path.

    A path knowledge_to_repo() refuses comes back unchanged.

    Examples:
        "myapp/myapp/config/settings_py/settings.py" -> "myapp/config/settings.py"
        "myapp/config/settings.py"                   -> "myapp/config/settings.py"

    Args:
        path: A repository-relative path or a knowledge path.
        project_name: The directory name of PROJECT_DIR.

    Returns:
        A repository-relative path; the argument unchanged when it already is one.
    """
    try:
        return knowledge_to_repo(path, project_name)
    except ValueError:
        return path


# The variable that names to a run in a container where the target repository is on the
# machine that started it, one absolute path per line (host_path_of)
HOST_PROJECT_ENV = "CTLOOM_HOST_PROJECT"


def host_path_of(path: str) -> str:
    """Write a path under the target repository's place on the starting machine from the root.

    Examples:
        ("/home/me/myapp/src/a.py") (HOST_PROJECT_ENV /home/me/myapp) -> "src/a.py"
        ("/home/me/myapp") (HOST_PROJECT_ENV /home/me/myapp)        -> "."
        ("/etc/hosts") or "src/a.py"                                -> as it is

    Args:
        path: A path.

    Returns:
        The path relative to the first directory HOST_PROJECT_ENV names that it is at
        or under, for an absolute path; path as it is otherwise, and when the variable
        is not set.
    """
    if not os.path.isabs(path):
        return path
    normal = os.path.normpath(path)
    for host in os.environ.get(HOST_PROJECT_ENV, "").split("\n"):
        if os.path.isabs(host) and os.path.commonpath([host, normal]) == host:
            return os.path.relpath(normal, host)
    return path


def to_repo_relative(path: str) -> str:
    """Convert whatever notation a path arrived in into a repository-relative one.

    A knowledge path is converted, an absolute path under the repository's place on the
    machine that started the run is written from there (host_path_of), an absolute path
    inside PROJECT_DIR is made relative to it, and anything else is normalised: an
    absolute path outside PROJECT_DIR, an absolute path the system cannot read as one,
    and every other path.

    Examples:
        ("./a.py") -> "a.py"
        ("/home/me/myapp/a.py") (PROJECT_DIR /home/me/myapp) -> "a.py"
        ("myapp/myapp/a_py/a.py") (project name myapp) -> "a.py"

    Args:
        path: A path in any of those notations.

    Returns:
        The repository-relative path.
    """
    if not path:
        return path
    name = os.path.basename(os.path.abspath(config.PROJECT_DIR))
    try:
        return knowledge_to_repo(path, name)
    except ValueError:
        pass
    path = host_path_of(path)
    if os.path.isabs(path):
        root = os.path.realpath(config.PROJECT_DIR)
        try:
            target = os.path.realpath(path)
        except ValueError:
            return os.path.normpath(path)
        if os.path.commonpath([root, target]) == root:
            return os.path.relpath(target, root)
        return os.path.normpath(path)
    return os.path.normpath(path)


def list_text(item_list: list[str]) -> str:
    """Render a few items inline, with a count standing in for the rest.

    How many items are named comes from config.MAX_NAME_ITEM.

    Examples:
        ["a.py", "b.py"]                        -> "a.py, b.py"
        ["a", "b", "c", "d", "e"] (MAX_NAME_ITEM 4) -> "a, b, c, d and 1 more"

    Args:
        item_list: The items to name, in the order they should appear.

    Returns:
        The rendered line, or "" for an empty list.
    """
    if not item_list:
        return ""
    rest_count = len(item_list) - config.MAX_NAME_ITEM
    head = ", ".join(item_list[:config.MAX_NAME_ITEM])
    return f"{head} and {rest_count} more" if rest_count > 0 else head


def test_root() -> str:
    """Return the directory config.TEST_PATH names, or "" when it names none.

    Examples:
        (TEST_PATH "tests") -> "tests"
        (TEST_PATH ".")     -> ""
        (TEST_PATH "")      -> ""

    Returns:
        The setting, or "" for an empty setting and for ".".
    """
    return config.TEST_PATH if config.TEST_PATH not in ("", ".") else ""


def _is_under_alt_test_root(path: str) -> bool:
    """Say whether a path is under one of config.TEST_PATH_ALT_TUPLE at the repository root.

    Examples:
        ("tests/fixtures/app.ts") (TEST_PATH_ALT_TUPLE ("tests",)) -> True
        ("pkg/tests/app.ts") (TEST_PATH_ALT_TUPLE ("tests",))      -> False

    Args:
        path: A repository-relative path, "/" between its parts.

    Returns:
        True when one of the directories is the path itself or its first part.
    """
    return any(path == root or path.startswith(f"{root}/")
               for root in config.TEST_PATH_ALT_TUPLE)


def has_test_rule() -> bool:
    """Say whether any path can be a test.

    Returns:
        True when a directory is taken for the tests (test_root(),
        config.TEST_PATH_ALT_TUPLE) or the primary language has a test_name_tuple.
    """
    return bool(test_root() or config.TEST_PATH_ALT_TUPLE
                or LANG_DICT[config.PRIMARY_EXT].test_name_tuple)


def has_name_of(part_iter: Iterable[str], pattern_set: Set[str]) -> bool:
    """Say whether one of some path parts matches one of some fnmatch patterns.

    Examples:
        (["src", "calc.egg-info"], {"*.egg-info", "obj"}) -> True
        (["src", "calc"], {"*.egg-info", "obj"})          -> False

    Args:
        part_iter: The parts, each a directory or file name.
        pattern_set: The patterns, matched case-sensitively; a name with no "*", "?"
            or "[" matches itself.

    Returns:
        True when a part matches a pattern.
    """
    return any(fnmatch.fnmatchcase(part, pattern)
               for part in part_iter for pattern in pattern_set)


def is_test_path(path: str, test_dir: str, name_tuple: tuple[str, ...]) -> bool:
    """Say whether a path is a test by a test directory and test name patterns.

    Examples:
        ("core/tests/test_a.py", "tests", ())         -> True
        ("src/__tests__/cfg.js", "", ("__tests__",))  -> True
        ("main.py", "tests", ("test_*.py",))          -> False

    Args:
        path: A path, "/" between its parts.
        test_dir: The test directory, matched as the path itself or one of its
            directories at any depth. "" names none.
        name_tuple: fnmatch patterns, matched against every part of the path.

    Returns:
        True when test_dir matches or a pattern matches a part.
    """
    return (bool(test_dir) and f"/{test_dir}/" in f"/{path}/") \
        or any(fnmatch.fnmatchcase(part, name)
               for part in path.split("/") for name in name_tuple)


def under_test_path(path_iter: Iterable[str]) -> list[str]:
    """Return the paths that are tests, in the order they came.

    A path is a test when config.TEST_PATH is the path itself or one of its directories,
    at the repository root or under any directory, when one of its parts matches one of
    the primary language's test_name_tuple (is_test_path()), or when it is under one of
    config.TEST_PATH_ALT_TUPLE at the repository root (_is_under_alt_test_root()). An empty
    TEST_PATH and "." name no directory (test_root()).

    Examples:
        (["tests/test_a.py", "main.py"]) (TEST_PATH "tests") -> ["tests/test_a.py"]
        (["core/src/test/java/a/FooTest.java"]) (TEST_PATH "src/test/java")
            -> ["core/src/test/java/a/FooTest.java"]
        (["src/__tests__/cfg.js", "src/cfg.test.js", "src/cfg.js"]) (primary "js")
            -> ["src/__tests__/cfg.js", "src/cfg.test.js"]
        (["tests/fixtures/app.ts", "pkg/tests/a.ts"]) (TEST_PATH "test",
            TEST_PATH_ALT_TUPLE ("tests",)) -> ["tests/fixtures/app.ts"]

    Args:
        path_iter: Repository-relative paths.

    Returns:
        Those of them that are tests.
    """
    root = test_root()
    name_tuple = LANG_DICT[config.PRIMARY_EXT].test_name_tuple
    return [path for path in path_iter
            if is_test_path(path, root, name_tuple) or _is_under_alt_test_root(path)]


# Where each repository keeps git itself, asked once per repository root
_GIT_DIR_BY_ROOT_DICT: dict[str, str] = {}


def _git_dir_of(root: str) -> str:
    """Return the directory git keeps the repository itself in, asked once per root.

    Args:
        root: The repository, as an absolute path.

    Returns:
        gitcmd.git_dir()'s answer, or "" for a directory that is not a repository.
    """
    if root not in _GIT_DIR_BY_ROOT_DICT:
        _GIT_DIR_BY_ROOT_DICT[root] = gitcmd.git_dir(root)
    return _GIT_DIR_BY_ROOT_DICT[root]


def path_text(path: object) -> str:
    """Return a path as a string.

    Args:
        path: What was passed: a string, or an os.PathLike that stands for one.

    Returns:
        The path as a string.

    Raises:
        ValueError: When it is neither.
    """
    if isinstance(path, str):
        return path
    if isinstance(path, os.PathLike) and isinstance(os.fspath(path), str):
        return os.fspath(path)
    raise ValueError(f"a path must be a string, not {type(path).__name__}")


def is_git_dir_path(full_path: str) -> bool:
    """Say whether a path is the directory git keeps the repository itself in, or in it.

    Args:
        full_path: An absolute path, compared as it is written: its symbolic links
            are not resolved here.

    Returns:
        True for <PROJECT_DIR>/.git, for the directory gitcmd.git_dir() names, and for
        every path under either.
    """
    root = os.path.realpath(config.PROJECT_DIR)
    return any(git_dir and os.path.commonpath([git_dir, full_path]) == git_dir
               for git_dir in (os.path.join(root, ".git"), _git_dir_of(root)))


def resolve_in_project(path: str) -> str:
    """Make path absolute and confirm it stays inside PROJECT_DIR.

    An absolute path under the repository's place on the machine that started the run
    is read from there (host_path_of).

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        The normalised absolute path.

    Raises:
        ValueError: When the path is not one path_text() takes, or the normalised path
            is outside PROJECT_DIR, inside OUTPUT_DIR, or inside the directory git
            keeps the repository itself in.
    """
    # realpath resolves symlinks before the containment check
    root = os.path.realpath(config.PROJECT_DIR)
    target = os.path.realpath(os.path.join(root, host_path_of(path_text(path))))
    if os.path.commonpath([root, target]) != root:
        raise ValueError(f"outside the repository: {path}. Write the path relative to the "
                         f"repository root, e.g. "
                         f"\"{PATH_EXAMPLE_DICT[config.PRIMARY_EXT]}\"")
    # Reject a path inside OUTPUT_DIR, which the check above lets through when
    # OUTPUT_DIR sits inside PROJECT_DIR
    out_dir_abs = os.path.realpath(config.OUTPUT_DIR)
    if os.path.exists(out_dir_abs) and os.path.commonpath([out_dir_abs, target]) == out_dir_abs:
        raise ValueError(
            f"OUTPUT_DIR holds generated output, not the original source: {path}")
    # Reject a path inside the directory git keeps the repository itself in
    if is_git_dir_path(target):
        raise ValueError(f"inside the git directory: {path}")
    return target


def file_in_project(path: str) -> str:
    """Return the absolute path of a target repository file small enough to read whole.

    The path is one resolve_in_project() takes and tmpfile.check_file() confirms: a
    regular file inside PROJECT_DIR once its symbolic links are resolved.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        The normalised absolute path.

    Raises:
        FileNotFoundError: When there is no such file.
        OSError: When it is not a regular file.
        ValueError: When resolve_in_project() refuses the path, or the file is larger
            than tmpfile.READ_MAX_BYTE.
    """
    full_path = resolve_in_project(path)
    tmpfile.check_file(full_path)
    return full_path


def real_rel(path: str) -> str:
    """Return the repository-relative path of a path with its symbolic links resolved.

    Examples:
        ("link.py") (link.py -> real.py) -> "real.py"
        ("./pkg/a.py")                   -> "pkg/a.py"

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        The path of resolve_in_project()'s answer relative to PROJECT_DIR.

    Raises:
        ValueError: When resolve_in_project() refuses the path.
    """
    return os.path.relpath(resolve_in_project(path), os.path.realpath(config.PROJECT_DIR))
