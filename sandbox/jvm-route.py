#!/usr/bin/env python3
"""Print the JVM options that stand for the proxy variables and SSL_CERT_FILE.

Usage: ctloom-jvm-route <directory>

What is printed is one line for JAVA_TOOL_OPTIONS, empty when nothing is set:
- https.proxyHost / https.proxyPort from HTTPS_PROXY, and http.proxyHost /
  http.proxyPort from HTTP_PROXY. ALL_PROXY stands in for either one that is not set.
  A name in lower case is read before the same name in upper case. Only a proxy
  spoken to over plain HTTP is handed on (http://host:port, or host:port), without the
  user and the password its URL carries
- http.nonProxyHosts from NO_PROXY, when a proxy was handed on. An entry that is an
  address range (10.0.0.0/8) is not handed on
- javax.net.ssl.trustStore, its type and its password, naming a PKCS12 trust store
  written to <directory> from the certificates of the PEM file SSL_CERT_FILE names

A host or an entry that holds anything but letters, digits and ".-_:[]*" is not handed
on. What is not handed on is said on standard error, one line each. Exits 0 whatever
was handed on, and 2 when started without the directory.
"""

import os
import re
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping

# The program that writes the trust store, started with the java launcher
STORE_SOURCE = "/opt/ctloom-jvm/TrustStore.java"

# The trust store's file name in the directory it is written to, and its password
_STORE_NAME = "ctloom-truststore.p12"
_STORE_PASSWORD = "changeit"

# How long writing the trust store may take, in seconds
_STORE_TIMEOUT_SEC = 120

# The port of a proxy whose URL names none
_DEFAULT_PORT = 80

# What a host or a pattern handed to the JVM is made of
_HOST_RE = re.compile(r"[A-Za-z0-9._:\[\]*-]+")


def _say(text: str) -> None:
    """Write one line to standard error."""
    print(f"ctloom-jvm-route: {text}", file=sys.stderr)


def _value_of(env_dict: Mapping[str, str], name: str) -> str:
    """Return a proxy variable's value, the name in lower case read first.

    Args:
        env_dict: The environment.
        name: The variable's name in upper case.

    Returns:
        The value without the blanks around it, or "".
    """
    return (env_dict.get(name.lower()) or env_dict.get(name) or "").strip()


def proxy_address(value: str) -> tuple[str, int] | None:
    """Return the host and the port of a proxy spoken to over plain HTTP.

    Examples:
        "http://me:pw@proxy.corp.example:3128" -> ("proxy.corp.example", 3128)
        "proxy.corp.example:3128"              -> ("proxy.corp.example", 3128)
        "http://proxy.corp.example"            -> ("proxy.corp.example", 80)
        "socks5://proxy.corp.example:1080"     -> None
        "http://proxy.corp.example:port"       -> None
        'http://proxy"corp.example:3128'       -> None

    Args:
        value: The value of a proxy variable.

    Returns:
        (host, port). None when the value names another scheme than http, no host, a
        host _HOST_RE does not match whole, or a port that is not a number.
    """
    try:
        url_split = urllib.parse.urlsplit(value if "://" in value else f"http://{value}")
        host, port = url_split.hostname, url_split.port
    except ValueError:
        return None
    if url_split.scheme.lower() != "http" or not _HOST_RE.fullmatch(host or ""):
        return None
    return host, port or _DEFAULT_PORT


def skip_host_list(value: str) -> tuple[list[str], list[str]]:
    """Return the patterns of http.nonProxyHosts that stand for a NO_PROXY value.

    Examples:
        "db,.corp.example"          -> (["db", "*.db", "corp.example",
                                        "*.corp.example"], [])
        "corp.example:8443"         -> (["corp.example", "*.corp.example"], [])
        "10.1.2.3, *.b.example, *"  -> (["10.1.2.3", "*.b.example", "*"], [])
        "10.0.0.0/8,a.example"      -> (["a.example", "*.a.example"], ["10.0.0.0/8"])

    Args:
        value: The value of NO_PROXY: hosts separated by commas.

    Returns:
        (the patterns, the entries that gave none). An entry gives its own pattern,
        without the port after a host name, an IPv4 address or an IPv6 address in
        brackets and without a "." it starts with, and a host name also the pattern
        of the names under it. An entry _HOST_RE does not match whole, an address
        range among them, gives none.
    """
    pattern_list, drop_list = [], []
    for entry in (text.strip() for text in value.split(",")):
        if not entry:
            continue
        if not _HOST_RE.fullmatch(entry):
            drop_list.append(entry)
            continue
        host, _, port = entry.rpartition(":")
        if host and port.isdecimal() and (":" not in host or host.endswith("]")):
            entry = host
        entry = entry.lstrip(".")
        if not entry:
            drop_list.append(".")
        elif "*" in entry or ":" in entry or entry.replace(".", "").isdecimal():
            pattern_list.append(entry)
        else:
            pattern_list += [entry, f"*.{entry}"]
    return pattern_list, drop_list


def proxy_option_list(env_dict: Mapping[str, str]) -> list[str]:
    """Return the JVM options that stand for the proxy variables of an environment.

    Examples:
        {"HTTPS_PROXY": "http://proxy.corp.example:3128", "NO_PROXY": ".corp.example"}
            -> ["-Dhttps.proxyHost=proxy.corp.example", "-Dhttps.proxyPort=3128",
                "-Dhttp.nonProxyHosts=corp.example|*.corp.example"]

    Args:
        env_dict: The environment.

    Returns:
        The options, as the module's description lists them. Empty when no proxy is
        handed on.
    """
    option_list = []
    for kind, name in (("https", "HTTPS_PROXY"), ("http", "HTTP_PROXY")):
        value = _value_of(env_dict, name) or _value_of(env_dict, "ALL_PROXY")
        if not value:
            continue
        address = proxy_address(value)
        if address is None:
            _say(f"the proxy {name} or ALL_PROXY names is not handed to the JVM tools: "
                 f"they take a proxy written as http://host:port")
            continue
        if "@" in value:
            _say(f"the user and the password of the proxy {name} or ALL_PROXY names "
                 f"are not handed to the JVM tools")
        option_list += [f"-D{kind}.proxyHost={address[0]}",
                        f"-D{kind}.proxyPort={address[1]}"]
    if not option_list:
        return []
    pattern_list, drop_list = skip_host_list(_value_of(env_dict, "NO_PROXY"))
    if drop_list:
        _say(f"these entries of NO_PROXY are not handed to the JVM tools: "
             f"{', '.join(drop_list)}")
    if pattern_list:
        option_list.append(f"-Dhttp.nonProxyHosts={'|'.join(pattern_list)}")
    return option_list


def trust_option_list(ca_file: str, directory: str) -> list[str]:
    """Write the trust store of a PEM file and return the JVM options naming it.

    Args:
        ca_file: The PEM file of certificate authorities.
        directory: The directory the trust store is written to.

    Returns:
        The trust store's path, type and password as options. Empty, with a line on
        standard error, when the store could not be written or holds no certificate.
    """
    store_file = os.path.join(directory, _STORE_NAME)
    try:
        proc = subprocess.run(
            ["java", STORE_SOURCE, ca_file, store_file, _STORE_PASSWORD],
            capture_output=True, text=True, errors="replace", check=False,
            stdin=subprocess.DEVNULL, timeout=_STORE_TIMEOUT_SEC)
    except (OSError, subprocess.TimeoutExpired) as e:
        _say(f"no trust store was written from {ca_file}: {type(e).__name__}: {e}")
        return []
    count_text = proc.stdout.strip()
    if proc.returncode or not count_text.isdecimal() or not int(count_text):
        reason = (proc.stderr.strip().splitlines() or ["it holds no certificate"])[-1]
        _say(f"no trust store was written from {ca_file}: {reason}")
        return []
    return [f"-Djavax.net.ssl.trustStore={store_file}",
            "-Djavax.net.ssl.trustStoreType=PKCS12",
            f"-Djavax.net.ssl.trustStorePassword={_STORE_PASSWORD}"]


def main(argv: list[str]) -> int:
    """Print the options for JAVA_TOOL_OPTIONS on one line.

    Args:
        argv: The program's name and the directory the trust store is written to.

    Returns:
        0, or 2 when argv names no directory.
    """
    if len(argv) != 2:
        _say("usage: ctloom-jvm-route <directory>")
        return 2
    option_list = proxy_option_list(os.environ)
    ca_file = os.environ.get("SSL_CERT_FILE", "")
    if ca_file:
        option_list += trust_option_list(ca_file, argv[1])
    print(" ".join(option_list))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
