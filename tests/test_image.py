"""The container image: its toolchains, its mark, and the command that builds it."""

import os
import re
import subprocess

import pytest

from codetwine_loom import config, image, sandbox
from codetwine_loom.languages import LANG_DICT
from conftest import CERTIFICATE_PEM

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DOCKERFILE = os.path.join(_REPO_DIR, "Dockerfile")


def _dockerfile_text() -> str:
    with open(_DOCKERFILE, encoding="utf-8") as fp:
        return fp.read()


def _case_arm_list_list() -> list[list[tuple[str, ...]]]:
    """Return the arms of each case over TOOLCHAIN in the Dockerfile, as their patterns."""
    block_list = re.findall(r'case "\$toolchain" in(.*?)esac', _dockerfile_text(), re.S)
    assert len(block_list) == 2
    return [[tuple(found.split("|")) for found in
             re.findall(r"^\s*([a-z*][a-z|]*|\*)\)", block, re.M)]
            for block in block_list]


def _arm_key(name: str, arm_list_list: list[list[tuple[str, ...]]]) -> tuple[int, ...]:
    """Return which arm of each case a TOOLCHAIN value takes."""
    key_list = []
    for arm_list in arm_list_list:
        index = next((i for i, arm in enumerate(arm_list) if name in arm), None)
        if index is None:
            index = next(i for i, arm in enumerate(arm_list) if arm == ("*",))
        key_list.append(index)
    return tuple(key_list)


# == the toolchains =============================================

def test_every_extension_codetwine_reads_has_a_group():
    assert set(image.EXT_GROUP_DICT) == set(LANG_DICT)


def test_the_groups_are_the_ones_the_dockerfile_s_case_tells_apart():
    """Two extensions are in one group exactly when both cases take the same arm for
    them, and a group's own name takes its extensions' arms."""
    arm_list_list = _case_arm_list_list()
    for ext, group in image.EXT_GROUP_DICT.items():
        assert _arm_key(ext, arm_list_list) == _arm_key(group or "py", arm_list_list), ext
        for other, other_group in image.EXT_GROUP_DICT.items():
            assert (_arm_key(ext, arm_list_list) == _arm_key(other, arm_list_list)) \
                == (group == other_group), (ext, other)


def test_the_default_toolchain_is_every_group_as_the_dockerfile_s_default_is():
    group_list = sorted(set(image.EXT_GROUP_DICT.values()) - {""})
    assert image.DEFAULT_TOOLCHAIN.split() == group_list
    found = re.search(r'^ARG TOOLCHAIN="([^"]*)"$', _dockerfile_text(), re.M)
    assert found and found.group(1) == image.DEFAULT_TOOLCHAIN


def test_the_dependencies_go_in_before_the_toolchains_and_the_package_after_them():
    """The dependencies are installed before the TOOLCHAIN step, and the package is
    copied after it."""
    text = _dockerfile_text()
    dep_at = text.index("uv pip install --no-cache --system --compile-bytecode -r ")
    toolchain_at = text.index("\nARG TOOLCHAIN=")
    package_at = text.index("\nCOPY codetwine_loom ")
    assert dep_at < toolchain_at < package_at
    assert "COPY codetwine_loom" not in text[:toolchain_at]


# == the mark ====================================================

def _source_repo(path) -> str:
    """Write a repository of codetwine-loom with the files an image is built from."""
    for name, text in (
            ("Dockerfile", "FROM x\n"),
            (".dockerignore", "**/__pycache__/\n**/*.pyc\n**/.env\ntests/\n"
                              "out_m1*/\n!codetwine_loom/keep.pyc\n"),
            ("pyproject.toml", "[project]\n"), ("uv.lock", "lock\n"), ("uv.toml", "\n"),
            ("codetwine_loom/a.py", "x = 1\n"), ("sandbox/run.sh", "echo\n")):
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text, encoding="utf-8")
    return str(path)


def test_the_mark_leaves_out_what_the_dockerignore_keeps_out_of_the_image(tmp_path):
    repo = _source_repo(tmp_path)
    mark = image.source_mark(repo)
    for name in ("codetwine_loom/__pycache__/a.cpython-312.pyc", "codetwine_loom/b.pyc",
                 "codetwine_loom/.env", "tests/test_a.py", "README.md", "notes/x.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("y\n", encoding="utf-8")
    assert image.source_mark(repo) == mark
    # A path a "!" rule lets back in is in the image, and in the mark
    (tmp_path / "codetwine_loom" / "keep.pyc").write_text("k\n", encoding="utf-8")
    assert image.source_mark(repo) != mark


@pytest.mark.parametrize("name", [
    "Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "uv.toml",
    "codetwine_loom/a.py", "sandbox/run.sh"])
def test_the_mark_changes_with_any_file_the_image_is_built_from(tmp_path, name):
    repo = _source_repo(tmp_path)
    mark = image.source_mark(repo)
    with open(tmp_path / name, "a", encoding="utf-8") as fp:
        fp.write("# changed\n")
    assert image.source_mark(repo) != mark


def test_a_file_added_under_the_package_changes_the_mark(tmp_path):
    repo = _source_repo(tmp_path)
    mark = image.source_mark(repo)
    (tmp_path / "codetwine_loom" / "new.py").write_text("\n", encoding="utf-8")
    assert image.source_mark(repo) != mark


@pytest.mark.parametrize("pattern, path, is_out", [
    ("**/__pycache__/", "codetwine_loom/index/__pycache__/x.pyc", True),
    ("**/__pycache__/", "__pycache__", True),
    ("out_m1*/", "out_m12/a", True),
    ("*.egg-info/", "x.egg-info/PKG-INFO", True),
    ("*.egg-info/", "codetwine_loom/x.egg-info", False),
    ("tests/", "codetwine_loom/tests/a.py", False),
    ("**/.env", "codetwine_loom/.env.example", False),
])
def test_a_dockerignore_pattern_matches_as_the_engine_reads_it(pattern, path, is_out):
    rule_list = [(False, image._ignore_regex(pattern.strip("/")))]
    assert image._is_ignore(path, rule_list) is is_out


def test_this_repository_s_mark_leaves_out_its_byte_code():
    file_list = image._source_file_list(image.source_dir())
    assert "Dockerfile" in file_list and "codetwine_loom/image.py" in file_list
    assert not [name for name in file_list
                if "__pycache__" in name or name.endswith(".pyc")]


def test_the_dockerignore_lets_in_everything_the_mark_is_made_of():
    rule_list = image._ignore_rule_list(image.source_dir())
    assert [name for name in (*image._SOURCE_FILE_TUPLE, *image._SOURCE_DIR_TUPLE)
            if image._is_ignore(name, rule_list)] == []


def test_the_dockerfile_copies_only_what_the_mark_is_made_of():
    copy_list = [part for line in re.findall(r"^COPY (?!--from)(.+)$", _dockerfile_text(),
                                             re.M)
                 for part in line.split()[:-1]]
    assert copy_list
    assert set(copy_list) <= set(image._SOURCE_FILE_TUPLE) | set(image._SOURCE_DIR_TUPLE)


def test_the_labels_are_given_on_the_command_line_and_not_by_the_dockerfile():
    argv = image.build_argv("/usr/bin/podman", "codetwine-loom:java", "java", "m1",
                            "/opt/ctl")
    label_list = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--label"]
    assert label_list == [f"{image.SOURCE_LABEL}=m1", f"{image.TOOLCHAIN_LABEL}=java"]
    assert image.SOURCE_LABEL not in _dockerfile_text()
    assert f'{image.TOOLCHAIN_LABEL}="$TOOLCHAIN"' in _dockerfile_text()


def test_the_package_s_repository_is_the_one_with_the_dockerfile(monkeypatch, tmp_path):
    assert image.source_dir() == _REPO_DIR
    monkeypatch.setattr(image, "__file__", str(tmp_path / "pkg" / "image.py"))
    assert image.source_dir() == ""


# == the state ===================================================

@pytest.mark.parametrize("label_dict, state", [
    (None, image.ABSENT_STATE),
    ({}, image.STALE_STATE),
    ({image.SOURCE_LABEL: ""}, image.STALE_STATE),
    ({image.SOURCE_LABEL: "other"}, image.STALE_STATE),
    ({image.SOURCE_LABEL: "m1", image.TOOLCHAIN_LABEL: "java"}, image.CURRENT_STATE),
])
def test_an_image_is_missing_stale_or_current(label_dict, state):
    assert image.image_state(label_dict, "m1") == state


@pytest.mark.parametrize("text, label_dict", [
    ('{"a": "1", "b": "x"}\n', {"a": "1", "b": "x"}), ("null\n", {}), ("", {}),
    ("<no value>", {}), ('["a"]', {})])
def test_the_labels_are_read_off_the_inspect(text, label_dict):
    assert image.label_dict_of(text) == label_dict


# == the build ===================================================

def test_podman_is_given_no_proxy_variable(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")
    assert image.build_argv("/usr/bin/podman", "codetwine-loom:java", "java", "m1",
                            "/opt/ctl") == [
        "/usr/bin/podman", "build", "--build-arg", "TOOLCHAIN=java",
        "--label", "io.codetwine-loom.source=m1", "--label", "io.codetwine-loom.toolchain=java",
        "-t", "codetwine-loom:java", "/opt/ctl"]


def test_docker_is_given_the_proxy_variables_and_the_certificate_authorities(
        monkeypatch, tmp_path):
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    _no_system_bundle(monkeypatch, tmp_path)
    monkeypatch.setenv("HTTPS_PROXY", "http://me:pw@proxy.corp.example:3128")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    argv = image.build_argv("/usr/bin/docker", "codetwine-loom", "py", "m1", "/opt/ctl")
    assert argv == [
        "/usr/bin/docker", "build", "--build-arg", "TOOLCHAIN=py",
        "--label", "io.codetwine-loom.source=m1", "--label", "io.codetwine-loom.toolchain=py",
        "--build-arg", "HTTPS_PROXY",
        "--build-arg", "no_proxy", "--build-arg", f"CA_PEM={CERTIFICATE_PEM}",
        "-t", "codetwine-loom", "/opt/ctl"]
    pem_file = image.ca_pem_file()
    assert pem_file.startswith(str(tmp_path / "cache" / "codetwine-loom" / "ca-pem"))
    text = image.build_command_text("/usr/bin/docker", "codetwine-loom", "py", "m1",
                                    "/opt/ctl")
    assert text == (f'docker build --build-arg TOOLCHAIN=py --label io.codetwine-loom.source=m1 '
                    f'--label io.codetwine-loom.toolchain=py '
                    f'--build-arg HTTPS_PROXY --build-arg no_proxy '
                    f'--build-arg CA_PEM="$(cat {pem_file})" -t codetwine-loom /opt/ctl')
    assert "pw@" not in text


def _no_system_bundle(monkeypatch, tmp_path, *bundle_text: str) -> None:
    """Give Mozilla's bundle (certifi) the certificates bundle_text holds, and a cache
    under tmp_path."""
    import certifi
    path = tmp_path / "mozilla.pem"
    path.write_text("".join(bundle_text), encoding="utf-8")
    monkeypatch.setattr(certifi, "where", lambda: str(path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


_KEY_PEM = ("-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg"
            "\n-----END PRIVATE KEY-----\n")


def test_a_private_key_in_the_file_of_certificate_authorities_is_not_passed(monkeypatch,
                                                                           tmp_path):
    ca_file = tmp_path / "mitmproxy-ca.pem"
    ca_file.write_text(_KEY_PEM + CERTIFICATE_PEM, encoding="utf-8")
    _no_system_bundle(monkeypatch, tmp_path)
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    argv = image.build_argv("/usr/bin/podman", "codetwine-loom", "py", "m1", "/opt/ctl")
    assert f"CA_PEM={CERTIFICATE_PEM}" in argv
    assert "PRIVATE KEY" not in " ".join(argv)
    pem_file = image.ca_pem_file()
    assert "PRIVATE KEY" not in open(pem_file, encoding="utf-8").read()
    assert os.stat(pem_file).st_mode & 0o077 == 0


def test_the_certificates_of_mozillas_bundle_are_not_passed(monkeypatch, tmp_path):
    """A system bundle with the proxy's certificate added passes only that one."""
    other = CERTIFICATE_PEM.replace("T96MKuf8", "T96MKuf9")
    ca_file = tmp_path / "bundle.pem"
    ca_file.write_text(other + CERTIFICATE_PEM, encoding="utf-8")
    _no_system_bundle(monkeypatch, tmp_path, other)
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    argv = image.build_argv("/usr/bin/podman", "codetwine-loom", "py", "m1", "/opt/ctl")
    assert f"CA_PEM={CERTIFICATE_PEM}" in argv


def test_a_file_of_only_mozillas_certificates_passes_none(monkeypatch, tmp_path):
    ca_file = tmp_path / "bundle.pem"
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    _no_system_bundle(monkeypatch, tmp_path, CERTIFICATE_PEM)
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    assert image.ca_pem_file() == ""
    assert not any(arg.startswith("CA_PEM=") for arg in image.build_argv(
        "/usr/bin/podman", "codetwine-loom", "py", "m1", "/opt/ctl"))


def test_the_file_ssl_cert_file_names_is_passed_whatever_ssl_reads_by_default(
        monkeypatch, tmp_path):
    """SSL_CERT_FILE also changes the bundle ssl reads by default."""
    ca_file = tmp_path / "corp-ca.pem"
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    argv = image.build_argv("/usr/bin/podman", "codetwine-loom", "py", "m1", "/opt/ctl")
    assert f"CA_PEM={CERTIFICATE_PEM}" in argv


def test_a_key_after_a_certificate_cut_short_is_not_passed(monkeypatch, tmp_path):
    cut = CERTIFICATE_PEM.replace("-----END CERTIFICATE-----\n", "")
    ca_file = tmp_path / "broken.pem"
    ca_file.write_text(cut + _KEY_PEM + CERTIFICATE_PEM, encoding="utf-8")
    _no_system_bundle(monkeypatch, tmp_path)
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    argv = image.build_argv("/usr/bin/podman", "codetwine-loom", "py", "m1", "/opt/ctl")
    assert "PRIVATE KEY" not in " ".join(argv)
    assert f"CA_PEM={CERTIFICATE_PEM}" in argv


def test_certificates_longer_than_an_argument_takes_stop_the_build(monkeypatch, tmp_path):
    many = "".join(CERTIFICATE_PEM.replace("T96MKuf8", f"T96M{number:04d}")
                   for number in range(image.CA_PEM_MAX_BYTE // len(CERTIFICATE_PEM) + 2))
    ca_file = tmp_path / "bundle.pem"
    ca_file.write_text(many, encoding="utf-8")
    _no_system_bundle(monkeypatch, tmp_path)
    monkeypatch.setenv(config.CA_FILE_ENV, str(ca_file))
    with pytest.raises(ValueError, match="only the certificate authorities you add"):
        image.build_argv("/usr/bin/podman", "codetwine-loom", "py", "m1", "/opt/ctl")


def test_a_toolchain_of_several_groups_is_quoted_in_the_command():
    assert image.build_command_text("/usr/bin/podman", "codetwine-loom:c-js", "c js", "m1",
                                    "/opt/ctl") == (
        "podman build --build-arg 'TOOLCHAIN=c js' --label io.codetwine-loom.source=m1 "
        "--label 'io.codetwine-loom.toolchain=c js' "
        "-t codetwine-loom:c-js /opt/ctl")


def test_the_build_runs_the_command_with_no_credential_and_a_time_limit(monkeypatch):
    seen = {}
    monkeypatch.setenv("LLM_API_KEY", "secret")

    def run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 3)
    monkeypatch.setattr(sandbox.subprocess, "run", run)
    assert sandbox.build_image(["/usr/bin/podman", "build", "/opt/ctl"]) == 3
    assert seen["argv"] == ["/usr/bin/podman", "build", "/opt/ctl"]
    assert "LLM_API_KEY" not in seen["env"] and seen["stdin"] == subprocess.DEVNULL
    assert seen["timeout"] == config.IMAGE_BUILD_TIMEOUT_SEC
