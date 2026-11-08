"""The public surface of codetwine-loom.

The functions below are imported from their modules on first use: importing the package
reads no setting. SettingError, keep_cwd_off_path(), cwd_module_list() and
hold_env_file() are defined here, ahead of config.py, and can be used before config is
first imported.
"""

import importlib
import os
import sys


class SettingError(ValueError):
    """A setting holds a value that is not accepted; the message names the setting."""


# The name of the file the settings are read from, in the directory a command is run from
ENV_FILE_NAME = ".env"

# The name of the setup subcommand
SETUP_COMMAND = "setup"

# What a signal's number is added to, to give the exit status of a process stopped by
# that signal, as a shell gives it
SIGNAL_STATUS_BASE = 128

# The .env files hold_env_file() was called for, by absolute path
_HOLD_ENV_FILE_SET: set[str] = set()


def hold_env_file(path: str = "") -> None:
    """Hold a .env file, which is_env_file_hold() then says True for.

    Args:
        path: The file, as an absolute path. "" holds the one in the working directory.
    """
    _HOLD_ENV_FILE_SET.add(path or os.path.abspath(ENV_FILE_NAME))


def is_env_file_hold(path: str) -> bool:
    """Say whether hold_env_file() was called for a .env file.

    Args:
        path: The file, as an absolute path.

    Returns:
        True when it was.
    """
    return path in _HOLD_ENV_FILE_SET


def _cwd_unless_own() -> str:
    """Return the working directory when this package was not imported from it.

    Returns:
        Its real path, or "" when there is no working directory or it is the directory
        this package is in.
    """
    try:
        cwd = os.path.realpath(os.getcwd())
    except OSError:
        return ""
    own = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return "" if cwd == own else cwd


def keep_cwd_off_path() -> None:
    """Take the working directory out of the places this process imports modules from.

    An entry of sys.path that is "", "." or the working directory itself is removed,
    whoever put it there. The entry this package was imported from is kept.
    """
    cwd = _cwd_unless_own()
    if not cwd:
        return
    sys.path[:] = [entry for entry in sys.path
                   if not isinstance(entry, str) or os.path.realpath(entry or ".") != cwd]


def cwd_module_list() -> list[str]:
    """Return the modules this process has imported from the working directory.

    A module counts when its top-level name was found in the working directory itself:
    as a file there, or as a package whose directory is there. The module the
    interpreter was started on is left out, and nothing counts when this package was
    imported from the working directory.

    Returns:
        The top-level names, in order.
    """
    cwd = _cwd_unless_own()
    if not cwd:
        return []
    name_list = []
    for name, module in sorted(sys.modules.items()):
        if "." in name or name == "__main__":
            continue
        path = getattr(module, "__file__", None)
        place_list = [os.path.dirname(entry) for entry in getattr(module, "__path__", ())
                      if isinstance(entry, str)] \
            or ([os.path.dirname(path)] if isinstance(path, str) else [])
        if any(os.path.realpath(place) == cwd for place in place_list):
            name_list.append(name)
    return name_list


_EXPORT_DICT = {
    "search_scope": "codetwine_loom.search.scope",
    "search_def_list": "codetwine_loom.search.scope",
    "expand": "codetwine_loom.search.scope",
    "create_agent": "codetwine_loom.agent",
    "run_edit": "codetwine_loom.agent",
}

__all__ = ["search_scope", "search_def_list", "expand", "create_agent", "run_edit",
           "SettingError"]


def __getattr__(name: str) -> object:
    """Import one of the public functions from its module.

    config is imported ahead of the module.

    Args:
        name: The attribute asked for.

    Returns:
        The function.

    Raises:
        AttributeError: When name is not one of the public functions.
    """
    if name not in _EXPORT_DICT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    importlib.import_module("codetwine_loom.config")
    return getattr(importlib.import_module(_EXPORT_DICT[name]), name)
