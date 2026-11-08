"""The entry point of python -m codetwine_loom and of the codetwine-loom command."""

import os
import signal
import sys

from codetwine_loom import (SETUP_COMMAND, SIGNAL_STATUS_BASE, SettingError,
                            cwd_module_list, hold_env_file, keep_cwd_off_path)


def main() -> None:
    """Run the command line, stopping with one line on a refused setting or a stop signal.

    A process whose working directory cannot be read stops here with one line before
    anything else is done. A process that has imported a module from the working
    directory stops here with one line that names it (cwd_module_list). The working
    directory is then taken off sys.path before the command line and what it uses are
    imported (keep_cwd_off_path), and standard output and standard error are set up
    (progress.set_up_output) before anything is written to them. For setup the .env
    file in the working directory is held (hold_env_file) before the command line is
    imported.

    A command stopped by Ctrl-C, SIGTERM or SIGHUP ends here with "stopped by <signal>"
    on standard error and 128 plus the signal's number as its exit status: the signal
    cli.stop_number() names, or SIGINT for one stopped before the command line was
    imported.

    Raises:
        SystemExit: With a line naming why the working directory cannot be read, with
            the names of the modules imported from the working directory, with the
            message of the SettingError, which names the setting, or with the exit
            status of a command stopped by a signal.
    """
    try:
        os.getcwd()
    except OSError as e:
        raise SystemExit(f"not started: the working directory cannot be read "
                         f"({e.strerror}). Start it from a directory that is "
                         f"there") from None
    name_list = cwd_module_list()
    keep_cwd_off_path()
    from codetwine_loom import progress
    progress.set_up_output()
    if name_list:
        raise SystemExit(
            f"not started: this process imported {', '.join(name_list)} from the "
            f"working directory, {os.getcwd()}, and not from the Python installation. "
            "Start it with the codetwine-loom command, or from another directory")
    # For setup, .env is held before config.py is imported
    if sys.argv[1:2] == [SETUP_COMMAND]:
        hold_env_file()
    cli = None
    try:
        from codetwine_loom import cli
        cli.main()
    except SettingError as e:
        raise SystemExit(str(e)) from None
    except KeyboardInterrupt:
        number = cli.stop_number() if cli is not None else signal.SIGINT
        if sys.stderr is not None:
            print(f"stopped by {number.name}", file=sys.stderr)
        raise SystemExit(SIGNAL_STATUS_BASE + number) from None


if __name__ == "__main__":
    main()
