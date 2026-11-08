"""Tests for sandbox/jvm-route.py, which turns the proxy variables and SSL_CERT_FILE
into JVM options."""

import importlib.util
import pathlib
import shutil
import subprocess

import pytest

from conftest import CERTIFICATE_PEM as _CERTIFICATE

_SANDBOX_DIR = pathlib.Path(__file__).resolve().parent.parent / "sandbox"

# The same certificate as openssl writes it with trust settings after it
_TRUST_CERTIFICATE = _CERTIFICATE.replace("CERTIFICATE-----", "TRUSTED CERTIFICATE-----").replace(
    "T96MKuf8\n", "T96MKuf8MBEwCgYIKwYBBQUHAwEMA2dsYQ==\n")


@pytest.fixture(scope="module")
def jvm_route():
    """The script, loaded as a module."""
    spec = importlib.util.spec_from_file_location("jvm_route", _SANDBOX_DIR / "jvm-route.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value, want", [
    ("http://me:pw@proxy.corp.example:3128", ("proxy.corp.example", 3128)),
    ("proxy.corp.example:3128", ("proxy.corp.example", 3128)),
    ("HTTP://Proxy.Corp.Example:8080/", ("proxy.corp.example", 8080)),
    ("http://proxy.corp.example", ("proxy.corp.example", 80)),
    ("http://[fd00::1]:3128", ("fd00::1", 3128)),
    ("https://proxy.corp.example:3128", None),
    ("socks5://proxy.corp.example:1080", None),
    ("http://proxy.corp.example:port", None),
    ("http://:3128", None),
    ("http://", None),
    ('http://proxy"corp.example:3128', None),
    ("http://proxy'corp.example:3128", None),
    ("http://proxy corp.example:3128", None),
    ("http://proxy$corp.example:3128", None)])
def test_a_proxy_spoken_to_over_plain_http_gives_its_host_and_port(jvm_route, value, want):
    assert jvm_route.proxy_address(value) == want


@pytest.mark.parametrize("value, want", [
    ("db,.corp.example", (["db", "*.db", "corp.example", "*.corp.example"], [])),
    ("..corp.example,.,.10.1.2.3", (["corp.example", "*.corp.example", "10.1.2.3"], ["."])),
    ("corp.example:8443", (["corp.example", "*.corp.example"], [])),
    (" 10.1.2.3 , *.b.example ,, * ", (["10.1.2.3", "*.b.example", "*"], [])),
    ("10.1.2.3:8443", (["10.1.2.3"], [])),
    ("[fd00::1]:8443,fd00::1:2", (["[fd00::1]", "fd00::1:2"], [])),
    ("10.0.0.0/8,a b,a|b,a.example", (["a.example", "*.a.example"],
                                      ["10.0.0.0/8", "a b", "a|b"])),
    ('x"y,' + "a'b,a$b,a\\b", ([], ['x"y', "a'b", "a$b", "a\\b"])),
    ("", ([], []))])
def test_the_hosts_reached_without_the_proxy_become_jvm_patterns(jvm_route, value, want):
    assert jvm_route.skip_host_list(value) == want


def test_no_option_holds_a_blank_or_a_quote_whatever_the_variables_hold(jvm_route):
    """The JVM splits JAVA_TOOL_OPTIONS at blanks and reads quotes in it."""
    for value in ('http://a"b:80', "http://a'b:80", "http://a b:80", "http://ok.example:80"):
        option_list = jvm_route.proxy_option_list(
            {"HTTPS_PROXY": value, "HTTP_PROXY": "http://ok.example:80",
             "NO_PROXY": 'x"y, a\'b ,ok.example,a b'})
        assert option_list[-1] == "-Dhttp.nonProxyHosts=ok.example|*.ok.example"
        assert not any(char in option for option in option_list for char in " \t\"'")


def test_the_entries_of_no_proxy_that_are_not_handed_on_are_said(jvm_route, capsys):
    assert jvm_route.proxy_option_list({
        "HTTPS_PROXY": "http://proxy.corp.example:3128",
        "NO_PROXY": "10.0.0.0/8,.corp.example,172.16.0.0/12"}) == [
            "-Dhttps.proxyHost=proxy.corp.example", "-Dhttps.proxyPort=3128",
            "-Dhttp.nonProxyHosts=corp.example|*.corp.example"]
    assert capsys.readouterr().err == (
        "ctloom-jvm-route: these entries of NO_PROXY are not handed to the JVM tools: "
        "10.0.0.0/8, 172.16.0.0/12\n")


def test_each_proxy_variable_gives_the_options_of_its_own_scheme(jvm_route):
    assert jvm_route.proxy_option_list({
        "HTTPS_PROXY": "http://proxy.corp.example:3128", "HTTP_PROXY": "other.example:8080",
        "NO_PROXY": "*.corp.example,10.1.2.3"}) == [
            "-Dhttps.proxyHost=proxy.corp.example", "-Dhttps.proxyPort=3128",
            "-Dhttp.proxyHost=other.example", "-Dhttp.proxyPort=8080",
            "-Dhttp.nonProxyHosts=*.corp.example|10.1.2.3"]


def test_a_name_in_lower_case_is_read_before_the_one_in_upper_case(jvm_route):
    assert jvm_route.proxy_option_list({
        "https_proxy": "http://lower.example:1", "HTTPS_PROXY": "http://upper.example:2",
        "no_proxy": "a.example", "NO_PROXY": "b.example"}) == [
            "-Dhttps.proxyHost=lower.example", "-Dhttps.proxyPort=1",
            "-Dhttp.nonProxyHosts=a.example|*.a.example"]


def test_all_proxy_stands_in_for_the_one_that_is_not_set(jvm_route):
    assert jvm_route.proxy_option_list({
        "ALL_PROXY": "http://all.example:3128", "HTTP_PROXY": "http://plain.example:8080"}) == [
            "-Dhttps.proxyHost=all.example", "-Dhttps.proxyPort=3128",
            "-Dhttp.proxyHost=plain.example", "-Dhttp.proxyPort=8080"]


def test_with_no_proxy_variable_nothing_is_handed_on(jvm_route, capsys):
    assert jvm_route.proxy_option_list({"NO_PROXY": "localhost", "PATH": "/bin"}) == []
    assert capsys.readouterr().err == ""


def test_a_proxy_of_another_scheme_is_said_and_not_handed_on(jvm_route, capsys):
    assert jvm_route.proxy_option_list({
        "HTTPS_PROXY": "socks5://ctloom-socks.corp.example:1080", "NO_PROXY": "localhost"}) == []
    err = capsys.readouterr().err
    assert err == ("ctloom-jvm-route: the proxy HTTPS_PROXY or ALL_PROXY names is not handed "
                   "to the JVM tools: they take a proxy written as http://host:port\n")


def test_the_user_and_the_password_of_a_proxy_are_said_and_not_handed_on(jvm_route, capsys):
    option_list = jvm_route.proxy_option_list(
        {"HTTPS_PROXY": "http://puser:proxy-pw-123@proxy.corp.example:3128"})
    assert option_list == ["-Dhttps.proxyHost=proxy.corp.example", "-Dhttps.proxyPort=3128"]
    err = capsys.readouterr().err
    assert "the user and the password of the proxy HTTPS_PROXY" in err
    assert "puser" not in err and "proxy-pw-123" not in err


def _java_answer(monkeypatch, jvm_route, returncode: int = 0, stdout: str = "",
                 stderr: str = "") -> list[list[str]]:
    """Have the java launcher answer as given, and return the commands it was run with."""
    argv_list = []

    def run(argv, **kwargs):
        argv_list.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
    monkeypatch.setattr(jvm_route.subprocess, "run", run)
    return argv_list


def test_the_trust_store_is_written_to_the_directory_and_named_in_the_options(
        jvm_route, monkeypatch):
    argv_list = _java_answer(monkeypatch, jvm_route, stdout="3\n")
    assert jvm_route.trust_option_list("/etc/ctloom-ca.pem", "/tmp") == [
        "-Djavax.net.ssl.trustStore=/tmp/ctloom-truststore.p12",
        "-Djavax.net.ssl.trustStoreType=PKCS12",
        "-Djavax.net.ssl.trustStorePassword=changeit"]
    assert argv_list == [["java", jvm_route.STORE_SOURCE, "/etc/ctloom-ca.pem",
                          "/tmp/ctloom-truststore.p12", "changeit"]]


@pytest.mark.parametrize("returncode, stdout, stderr, reason", [
    (1, "", "Exception in thread\njava.io.FileNotFoundException: no.pem",
     "java.io.FileNotFoundException: no.pem"),
    (0, "0\n", "", "it holds no certificate"),
    (0, "", "", "it holds no certificate")])
def test_a_trust_store_that_could_not_be_written_is_said_and_gives_no_option(
        jvm_route, monkeypatch, capsys, returncode, stdout, stderr, reason):
    _java_answer(monkeypatch, jvm_route, returncode, stdout, stderr)
    assert jvm_route.trust_option_list("/etc/ctloom-ca.pem", "/tmp") == []
    assert capsys.readouterr().err == (
        f"ctloom-jvm-route: no trust store was written from /etc/ctloom-ca.pem: {reason}\n")


def test_without_a_java_launcher_no_option_names_a_trust_store(jvm_route, monkeypatch,
                                                              capsys):
    def run(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "java")
    monkeypatch.setattr(jvm_route.subprocess, "run", run)
    assert jvm_route.trust_option_list("/etc/ctloom-ca.pem", "/tmp") == []
    assert "no trust store was written from /etc/ctloom-ca.pem: FileNotFoundError" in (
        capsys.readouterr().err)


def test_the_options_are_printed_on_one_line(jvm_route, monkeypatch, capsys):
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ctloom-ca.pem")
    _java_answer(monkeypatch, jvm_route, stdout="1\n")
    assert jvm_route.main(["ctloom-jvm-route", "/tmp"]) == 0
    assert capsys.readouterr().out == (
        "-Dhttps.proxyHost=proxy.corp.example -Dhttps.proxyPort=3128 "
        "-Djavax.net.ssl.trustStore=/tmp/ctloom-truststore.p12 "
        "-Djavax.net.ssl.trustStoreType=PKCS12 "
        "-Djavax.net.ssl.trustStorePassword=changeit\n")


def test_started_without_the_directory_it_says_how_it_is_started(jvm_route, capsys):
    assert jvm_route.main(["ctloom-jvm-route"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage: ctloom-jvm-route <directory>" in captured.err


def test_with_nothing_set_an_empty_line_is_printed_and_java_is_not_started(
        jvm_route, monkeypatch, capsys):
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    argv_list = _java_answer(monkeypatch, jvm_route)
    assert jvm_route.main(["ctloom-jvm-route", "/tmp"]) == 0
    assert capsys.readouterr().out == "\n" and argv_list == []


@pytest.mark.skipif(shutil.which("java") is None or shutil.which("keytool") is None,
                    reason="needs a JDK on PATH")
def test_the_trust_store_holds_every_certificate_of_the_file(jvm_route, monkeypatch,
                                                            tmp_path):
    """The file starts with a byte order mark and holds two certificates, one of them
    with CRLF line ends and the other with trust settings after it, with lines of text
    and the block of a key around them."""
    key_block = "-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"
    assert "TRUSTED CERTIFICATE" in _TRUST_CERTIFICATE and "MA2dsYQ==" in _TRUST_CERTIFICATE
    ca_file = tmp_path / "ca.pem"
    ca_file.write_bytes(
        b"\xef\xbb\xbf" + f"# one\n{key_block}{_CERTIFICATE}\n# the same once more\n"
        f"{_TRUST_CERTIFICATE}".replace("\n", "\r\n", 12).encode("utf-8"))
    monkeypatch.setattr(jvm_route, "STORE_SOURCE", str(_SANDBOX_DIR / "TrustStore.java"))
    option_list = jvm_route.trust_option_list(str(ca_file), str(tmp_path))
    assert option_list[0] == f"-Djavax.net.ssl.trustStore={tmp_path / 'ctloom-truststore.p12'}"
    proc = subprocess.run(
        ["keytool", "-list", "-keystore", str(tmp_path / "ctloom-truststore.p12"),
         "-storepass", "changeit"], capture_output=True, text=True, check=False)
    assert proc.stdout.count("trustedCertEntry") == 2, proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("java") is None, reason="needs a JDK on PATH")
def test_a_file_that_holds_no_certificate_is_said_in_one_line(jvm_route, monkeypatch,
                                                             tmp_path, capsys):
    monkeypatch.setattr(jvm_route, "STORE_SOURCE", str(_SANDBOX_DIR / "TrustStore.java"))
    for text, reason in (("", "it holds no certificate"),
                         ("-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n",
                          "java.security.cert.CertificateException")):
        ca_file = tmp_path / "ca.pem"
        ca_file.write_text(text, encoding="utf-8")
        assert jvm_route.trust_option_list(str(ca_file), str(tmp_path)) == []
        err = capsys.readouterr().err
        assert err.startswith(f"ctloom-jvm-route: no trust store was written from {ca_file}: "
                              f"{reason}") and err.count("\n") == 1, err
