"""The lock taken when the writers of one thing are not all in one process.

What it is for is exercised where it is used - two runs recording into one history
(test_history.py) and two runs taking a patch in to one repository (test_sandbox.py).
What is here is the file it makes and what it does where there is no lock to take.
"""

import pathlib
import threading

from codetwine_loom import filelock


def test_the_lock_sits_beside_what_it_stands_for(tmp_path):
    """The history and the patch are moved into place rather than written where they
    stand, and a lock on a file that is replaced is a lock on nothing."""
    thing = tmp_path / "edit_history.json"
    with filelock.hold_lock(str(thing), "the thing"):
        assert pathlib.Path(filelock.path_for(str(thing))).exists()
    assert not thing.exists()


def test_a_system_with_no_file_locks_says_so_and_carries_on(tmp_path, monkeypatch,
                                                            caplog):
    """Nobody should read a run that went without a lock as one that held one."""
    monkeypatch.setattr(filelock, "fcntl", None)
    ran = False
    with filelock.hold_lock(str(tmp_path / "thing"), "the thing"):
        ran = True
    assert ran and "no file locks" in caplog.text


def test_a_lock_that_cannot_be_made_is_reported_and_not_raised(tmp_path, caplog):
    """A directory that is not there is not a reason to stop what was being written."""
    ran = False
    with filelock.hold_lock(str(tmp_path / "nowhere" / "thing"), "the thing"):
        ran = True
    assert ran and "without its lock" in caplog.text


def test_a_lock_a_thread_already_holds_is_not_waited_for(tmp_path):
    """The analysis is made under the lock its copies are made under, and the first
    analysis of an output directory is made from inside that same lock."""
    target = str(tmp_path / "thing")
    with filelock.hold_lock(target, "the outer block"):
        with filelock.hold_lock(target, "the inner block"):
            assert True


def test_two_threads_wait_for_one_another(tmp_path):
    """The agents a dispatching run starts at once are threads of one process."""
    target = str(tmp_path / "thing")
    order = []

    def second():
        with filelock.hold_lock(target, "the second thread"):
            order.append("second")

    with filelock.hold_lock(target, "the first thread"):
        thread = threading.Thread(target=second)
        thread.start()
        thread.join(timeout=0.3)
        order.append("first")
    thread.join(timeout=5)
    assert order == ["first", "second"]


def test_a_lock_another_holds_is_waited_for_and_said_in_the_log(tmp_path, caplog):
    target = str(tmp_path / "thing")
    caplog.set_level("INFO")

    def second():
        with filelock.hold_lock(target, "the thing"):
            pass

    with filelock.hold_lock(target, "the thing"):
        thread = threading.Thread(target=second)
        thread.start()
        thread.join(timeout=0.3)
    thread.join(timeout=5)
    assert "waiting for the thing, which another run of this tool holds" in caplog.text


def test_a_free_lock_is_taken_without_a_word(tmp_path, caplog):
    caplog.set_level("INFO")
    with filelock.hold_lock(str(tmp_path / "thing"), "the thing"):
        pass
    assert "waiting" not in caplog.text


def test_a_lock_another_holds_is_not_waited_for_when_told_not_to(tmp_path):
    target = str(tmp_path / "thing")
    seen = []

    def second():
        try:
            with filelock.hold_lock(target, "the thing", wait=False):
                seen.append("ran")
        except filelock.LockBusy as e:
            seen.append(str(e))

    with filelock.hold_lock(target, "the thing"):
        thread = threading.Thread(target=second)
        thread.start()
        thread.join(timeout=5)
    assert seen == ["another run of this tool holds the thing"]
    thread = threading.Thread(target=second)
    thread.start()
    thread.join(timeout=5)
    assert seen[-1] == "ran"
