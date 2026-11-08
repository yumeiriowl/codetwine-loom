"""Tests for sandbox/gradle-share.py, which keeps the dependencies Gradle fetched where
every container reads them."""

import importlib.util
import os
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "gradle-share.py"


@pytest.fixture(scope="module")
def gradle_share():
    """The script, loaded as a module."""
    spec = importlib.util.spec_from_file_location("gradle_share", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _home_with(home: pathlib.Path, file_dict: dict[str, str]) -> pathlib.Path:
    """Make a Gradle home whose dependency directory holds the files given."""
    for rel, text in file_dict.items():
        path = home / "caches" / "modules-2" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return home


def _tree(directory: pathlib.Path) -> dict[str, str]:
    """Return {path under directory: content} of the files in it, links followed."""
    return {os.path.relpath(os.path.join(root, name), directory):
            pathlib.Path(root, name).read_text(encoding="utf-8")
            for root, _, name_list in os.walk(directory, followlinks=True)
            for name in name_list}


def test_what_a_home_fetched_is_read_under_one_name_in_the_share(gradle_share, tmp_path):
    home = _home_with(tmp_path / "home", {
        "files-2.1/g/a/1/hash/a.jar": "jar", "metadata-2.106/x.bin": "bin",
        "modules-2.lock": "lock", "files-2.1/files-2.1.lock": "lock",
        "gc.properties": "gc"})
    share = tmp_path / "cache" / "gradle"
    assert gradle_share.publish(str(home), str(share)) is True
    assert _tree(share / "modules-2") == {
        "files-2.1/g/a/1/hash/a.jar": "jar", "metadata-2.106/x.bin": "bin"}
    assert (share / "modules-2").is_symlink()
    assert not os.path.isabs(os.readlink(share / "modules-2"))
    assert sorted(os.listdir(share)) == sorted(
        ["modules-2", os.readlink(share / "modules-2")])


def test_a_later_publish_replaces_what_the_name_points_at_and_removes_the_older_copy(
        gradle_share, tmp_path):
    share = tmp_path / "gradle"
    gradle_share.publish(str(_home_with(tmp_path / "one", {"files-2.1/a.jar": "1"})),
                         str(share))
    first = os.readlink(share / "modules-2")
    (share / "snap.left-half-written").mkdir()
    (share / "modules-2.new").symlink_to("snap.left-half-written")
    gradle_share.publish(
        str(_home_with(tmp_path / "two", {"files-2.1/a.jar": "1", "files-2.1/b.jar": "2"})),
        str(share))
    second = os.readlink(share / "modules-2")
    assert second != first and sorted(os.listdir(share)) == sorted(["modules-2", second])
    assert _tree(share / "modules-2") == {"files-2.1/a.jar": "1", "files-2.1/b.jar": "2"}


def test_a_directory_under_the_name_is_replaced_by_the_link(gradle_share, tmp_path):
    share = tmp_path / "gradle"
    (share / "modules-2" / "files-2.1").mkdir(parents=True)
    (share / "modules-2" / "files-2.1" / "old.jar").write_text("old", encoding="utf-8")
    gradle_share.publish(str(_home_with(tmp_path / "home", {"files-2.1/a.jar": "1"})),
                         str(share))
    assert (share / "modules-2").is_symlink()
    assert _tree(share / "modules-2") == {"files-2.1/a.jar": "1"}


def test_a_home_that_fetched_nothing_leaves_the_share_as_it_is(gradle_share, tmp_path):
    share = tmp_path / "gradle"
    gradle_share.publish(str(_home_with(tmp_path / "one", {"files-2.1/a.jar": "1"})),
                         str(share))
    before = os.readlink(share / "modules-2")
    (tmp_path / "empty" / "caches").mkdir(parents=True)
    assert gradle_share.publish(str(tmp_path / "empty"), str(share)) is False
    assert gradle_share.publish(str(tmp_path / "no-such-home"), str(share)) is False
    assert os.readlink(share / "modules-2") == before
    assert gradle_share.publish(str(tmp_path / "empty"), str(tmp_path / "never")) is False
    assert not (tmp_path / "never").exists()


def test_a_home_is_started_from_what_the_share_holds(gradle_share, tmp_path):
    share = tmp_path / "gradle"
    gradle_share.publish(str(_home_with(tmp_path / "one", {"files-2.1/a.jar": "1"})),
                         str(share))
    home = tmp_path / "run" / "gradle-home"
    assert gradle_share.seed(str(share), str(home)) is True
    target = home / "caches" / "modules-2"
    assert _tree(target) == {"files-2.1/a.jar": "1"} and not target.is_symlink()
    # What the home writes afterwards does not reach the share
    (target / "files-2.1" / "b.jar").write_text("2", encoding="utf-8")
    (target / "files-2.1" / "a.jar").write_text("changed", encoding="utf-8")
    assert _tree(share / "modules-2") == {"files-2.1/a.jar": "1"}
    # A home that already has the directory is left as it is
    assert gradle_share.seed(str(share), str(home)) is False
    assert _tree(target) == {"files-2.1/a.jar": "changed", "files-2.1/b.jar": "2"}


def test_a_home_that_could_not_be_started_whole_is_left_without_the_copy(
        gradle_share, tmp_path, capsys):
    """A home that holds part of the share would put less there than the share holds."""
    share = tmp_path / "gradle"
    gradle_share.publish(
        str(_home_with(tmp_path / "one", {"files-2.1/a.jar": "1", "files-2.1/b.jar": "2"})),
        str(share))
    unreadable = share / "modules-2" / "files-2.1" / "b.jar"
    unreadable.chmod(0)
    if os.access(unreadable, os.R_OK):
        pytest.skip("this user reads a file of mode 0")
    home = tmp_path / "run" / "gradle-home"
    assert gradle_share.main(["ctloom-gradle-share", "seed", str(share), str(home)]) == 1
    assert not (home / "caches" / "modules-2").exists()
    err = capsys.readouterr().err
    assert err.startswith(f"ctloom-gradle-share: seed of {share} to {home} could not be "
                          f"done: ") and err.count("\n") == 1
    unreadable.chmod(0o644)
    assert _tree(share / "modules-2") == {"files-2.1/a.jar": "1", "files-2.1/b.jar": "2"}


@pytest.mark.parametrize("where", ["outside", "deeper", "the share itself"])
def test_a_name_that_leads_out_of_the_share_starts_no_home(gradle_share, tmp_path, capsys,
                                                          where):
    """Every container writes to the share, and a link in it can lead anywhere."""
    share = tmp_path / "gradle"
    (share / "snap.1" / "inner").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "big").write_text("x", encoding="utf-8")
    target = {"outside": tmp_path / "elsewhere", "deeper": share / "snap.1" / "inner",
              "the share itself": share}[where]
    (share / "modules-2").symlink_to(target)
    home = tmp_path / "home"
    assert gradle_share.main(["ctloom-gradle-share", "seed", str(share), str(home)]) == 0
    assert not (home / "caches" / "modules-2").exists()
    assert f"which is not a directory of {share}, so no home is started from it" in (
        capsys.readouterr().err)


def test_only_the_files_and_the_directories_are_copied_either_way(gradle_share, tmp_path):
    home = _home_with(tmp_path / "home", {"files-2.1/a.jar": "1"})
    module_dir = home / "caches" / "modules-2"
    (tmp_path / "outside").write_text("not a dependency", encoding="utf-8")
    (module_dir / "files-2.1" / "link").symlink_to(tmp_path / "outside")
    (module_dir / "dir-link").symlink_to("/")
    os.mkfifo(module_dir / "files-2.1" / "pipe")
    share = tmp_path / "gradle"
    assert gradle_share.publish(str(home), str(share)) is True
    assert _tree(share / "modules-2") == {"files-2.1/a.jar": "1"}
    snap = share / os.readlink(share / "modules-2")
    (snap / "files-2.1" / "link").symlink_to(tmp_path / "outside")
    os.mkfifo(snap / "pipe")
    other = tmp_path / "other"
    assert gradle_share.seed(str(share), str(other)) is True
    assert _tree(other / "caches" / "modules-2") == {"files-2.1/a.jar": "1"}


@pytest.mark.parametrize("make", ["nothing", "empty", "dangling"])
def test_a_share_that_holds_nothing_starts_no_home(gradle_share, tmp_path, make):
    share = tmp_path / "gradle"
    if make != "nothing":
        share.mkdir()
    if make == "dangling":
        (share / "modules-2").symlink_to("snap.gone")
    assert gradle_share.seed(str(share), str(tmp_path / "home")) is False
    assert not (tmp_path / "home").exists()


def test_the_two_forms_end_with_the_status_of_what_they_did(gradle_share, tmp_path, capsys):
    share = tmp_path / "gradle"
    home = _home_with(tmp_path / "home", {"files-2.1/a.jar": "1"})
    assert gradle_share.main(["ctloom-gradle-share", "publish", str(home), str(share)]) == 0
    assert gradle_share.main(["ctloom-gradle-share", "seed", str(share),
                              str(tmp_path / "other")]) == 0
    assert (tmp_path / "other" / "caches" / "modules-2" / "files-2.1" / "a.jar").exists()
    assert capsys.readouterr().err == ""
    for argv in (["ctloom-gradle-share"], ["ctloom-gradle-share", "seed", str(share)],
                 ["ctloom-gradle-share", "copy", str(share), str(home)]):
        assert gradle_share.main(argv) == 2
    assert capsys.readouterr().err.count("usage: ctloom-gradle-share") == 3


def test_a_share_that_cannot_be_written_is_said_in_one_line(gradle_share, tmp_path, capsys):
    home = _home_with(tmp_path / "home", {"files-2.1/a.jar": "1"})
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert gradle_share.main(["ctloom-gradle-share", "publish", str(home),
                              str(blocker / "gradle")]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"ctloom-gradle-share: publish of {home} to {blocker / 'gradle'} "
                          f"could not be done: ") and err.count("\n") == 1
