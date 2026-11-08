"""The container image a run starts its containers from: its mark and its build.

An image built here holds the toolchains of every language by default
(DEFAULT_TOOLCHAIN). It carries two labels: the mark of what this installation of
codetwine-loom puts into an image (source_mark) and the TOOLCHAIN it was built with. An
image whose mark is missing or is not the mark of this installation is stale
(image_state).
"""

import hashlib
import json
import logging
import os
import re
import shlex

from codetwine_loom import config

logger = logging.getLogger(__name__)

# == The toolchains =============================================
# The extensions of LANG_DICT by the toolchain group the image is given for them, as the
# case of the Dockerfile's TOOLCHAIN groups them. "" is the group whose files need no
# toolchain beyond what every image holds
_GROUP_EXT_DICT: dict[str, tuple[str, ...]] = {
    "": ("py", "sql"),
    "java": ("java",),
    "kt": ("kt", "kts"),
    "js": ("js", "jsx", "mjs", "cjs"),
    "ts": ("ts", "tsx", "mts", "cts"),
    "c": ("c", "cpp", "h", "cc", "cxx", "hpp", "hh", "hxx"),
    "rs": ("rs",),
    "cs": ("cs",),
    "cbl": ("cbl", "cob", "cpy", "bms", "jcl", "prc", "proc", "csd"),
    "r": ("r", "rmd", "qmd"),
}

# The toolchain group of each extension of LANG_DICT
EXT_GROUP_DICT: dict[str, str] = {ext: group for group, ext_tuple in _GROUP_EXT_DICT.items()
                                  for ext in ext_tuple}

# The TOOLCHAIN an image is built with unless one is named: every group, as the
# Dockerfile's TOOLCHAIN defaults to
DEFAULT_TOOLCHAIN = " ".join(sorted(group for group in _GROUP_EXT_DICT if group))

# == The labels =================================================
# The label holding the mark of what the image was built from (source_mark)
SOURCE_LABEL = "io.codetwine-loom.source"

# The label holding the TOOLCHAIN the image was built with
TOOLCHAIN_LABEL = "io.codetwine-loom.toolchain"

# What image_state() says of an image
ABSENT_STATE = "absent"
STALE_STATE = "stale"
CURRENT_STATE = "current"

# == What goes into an image ====================================
# The files and directories of codetwine-loom's repository the Dockerfile copies into an
# image, relative to the repository
_SOURCE_FILE_TUPLE = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "uv.toml")
_SOURCE_DIR_TUPLE = ("codetwine_loom", "sandbox")

# The file that keeps paths out of the build context, in the repository's root
_IGNORE_FILE = ".dockerignore"

# The build argument the Dockerfile reads the certificate authorities' PEM from
_CA_BUILD_ARG = "CA_PEM"

# A PEM block of one certificate, its body base64 and blank space only; nothing else of
# the file (a private key) is passed
_CERT_BLOCK_RE = re.compile(
    r"-----BEGIN CERTIFICATE-----[A-Za-z0-9+/=\s]*?-----END CERTIFICATE-----")

# The largest CA_PEM an image build is given (bytes), under the 128 KiB a process takes
# for one argument
CA_PEM_MAX_BYTE = 100_000


# Where the certificates CA_PEM passes are written, under the user's cache directory
_CA_PEM_DIR = os.path.join("codetwine-loom", "ca-pem")


def source_dir() -> str:
    """Return the repository of codetwine-loom this package was installed from.

    Returns:
        The directory above the package, as an absolute path, when it holds a
        Dockerfile; "" otherwise.
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return repo_dir if os.path.isfile(os.path.join(repo_dir, "Dockerfile")) else ""


def _ignore_regex(pattern: str) -> re.Pattern[str]:
    """Turn one pattern of a .dockerignore file into a regex over a relative path.

    Examples:
        "**/__pycache__/" matches "__pycache__" and "a/b/__pycache__"
        "build*/"         matches "build2"
        "*.egg-info"      matches "x.egg-info" and not "a/x.egg-info"

    Args:
        pattern: The pattern, without its "!" and with "/" at both ends taken off.

    Returns:
        The regex, to be matched against the whole path.
    """
    regex_text = ""
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            regex_text += "(?:.*/)?"
            index += 3
        elif pattern.startswith("**", index):
            regex_text += ".*"
            index += 2
        elif pattern[index] == "*":
            regex_text += "[^/]*"
            index += 1
        elif pattern[index] == "?":
            regex_text += "[^/]"
            index += 1
        elif pattern[index] == "[" and "]" in pattern[index + 1:]:
            end = pattern.index("]", index + 1)
            regex_text += "[" + pattern[index + 1:end].replace("\\", "\\\\") + "]"
            index = end + 1
        else:
            regex_text += re.escape(pattern[index])
            index += 1
    return re.compile(regex_text)


def _ignore_rule_list(repo_dir: str) -> list[tuple[bool, re.Pattern[str]]]:
    """Read the rules of the repository's .dockerignore.

    Args:
        repo_dir: codetwine-loom's repository, as source_dir() returns it.

    Returns:
        (whether the rule lets a path back in ("!"), its regex) for each pattern, in the
        file's order. Empty when there is no such file.
    """
    try:
        with open(os.path.join(repo_dir, _IGNORE_FILE), encoding="utf-8") as fp:
            line_list = list(fp)
    except FileNotFoundError:
        return []
    rule_list = []
    for line in line_list:
        pattern = line.strip()
        if not pattern or pattern.startswith("#"):
            continue
        is_back = pattern.startswith("!")
        pattern = pattern.lstrip("!").strip().strip("/")
        if pattern:
            rule_list.append((is_back, _ignore_regex(pattern)))
    return rule_list


def _is_ignore(file_rel: str, rule_list: list[tuple[bool, re.Pattern[str]]]) -> bool:
    """Say whether .dockerignore keeps a path out of the build context.

    A rule that matches the path or a directory above it decides, and the last one to
    match wins.

    Args:
        file_rel: The path relative to the repository, with "/" between its parts.
        rule_list: _ignore_rule_list()'s return value.

    Returns:
        True when the path is kept out.
    """
    part_list = file_rel.split("/")
    prefix_list = ["/".join(part_list[:count]) for count in range(1, len(part_list) + 1)]
    is_out = False
    for is_back, regex in rule_list:
        if any(regex.fullmatch(prefix) for prefix in prefix_list):
            is_out = not is_back
    return is_out


def _source_file_list(repo_dir: str) -> list[str]:
    """Return the files of the repository the Dockerfile copies into an image.

    Args:
        repo_dir: codetwine-loom's repository, as source_dir() returns it.

    Returns:
        Their paths relative to the repository, sorted, without the ones .dockerignore
        keeps out. Links are not followed.
    """
    rule_list = _ignore_rule_list(repo_dir)
    file_list = [name for name in _SOURCE_FILE_TUPLE
                 if os.path.isfile(os.path.join(repo_dir, name))]
    for top in _SOURCE_DIR_TUPLE:
        for dir_path, _, file_name_list in os.walk(os.path.join(repo_dir, top)):
            for file_name in file_name_list:
                path = os.path.join(dir_path, file_name)
                if os.path.isfile(path) and not os.path.islink(path):
                    file_list.append(os.path.relpath(path, repo_dir).replace(os.sep, "/"))
    return sorted(file_rel for file_rel in file_list
                  if not _is_ignore(file_rel, rule_list))


def source_mark(repo_dir: str) -> str:
    """Return the mark of what an image built from the repository holds.

    Args:
        repo_dir: codetwine-loom's repository, as source_dir() returns it.

    Returns:
        The sha256 hex digest of each file of _source_file_list() and its contents' own
        digest.
    """
    mark_hash = hashlib.sha256()
    for file_rel in _source_file_list(repo_dir):
        with open(os.path.join(repo_dir, file_rel), "rb") as fp:
            file_hash = hashlib.sha256(fp.read()).hexdigest()
        mark_hash.update(f"{file_rel}\0{file_hash}\n".encode("utf-8", "surrogateescape"))
    return mark_hash.hexdigest()


def label_dict_of(inspect_text: str) -> dict[str, str]:
    """Read an image's labels off what the engine's image inspect printed.

    Args:
        inspect_text: The JSON of the labels ("{{json .Config.Labels}}"), "null" when
            the image has none.

    Returns:
        {label: value}; empty when the text holds no object.
    """
    try:
        label_dict = json.loads(inspect_text or "null")
    except ValueError:
        logger.info("the image's labels could not be read: %s", inspect_text.strip())
        return {}
    if not isinstance(label_dict, dict):
        return {}
    return {str(key): str(value) for key, value in label_dict.items()}


def image_state(label_dict: dict[str, str] | None, mark: str) -> str:
    """Say whether an image is there and was built from this installation.

    Args:
        label_dict: The image's labels, or None when the engine does not have it.
        mark: source_mark() of this installation.

    Returns:
        ABSENT_STATE for None, CURRENT_STATE when SOURCE_LABEL holds mark, and
        STALE_STATE otherwise.
    """
    if label_dict is None:
        return ABSENT_STATE
    return CURRENT_STATE if label_dict.get(SOURCE_LABEL) == mark else STALE_STATE


def _build_part_tuple(engine_path: str, image_name: str, toolchain: str, mark: str,
                      repo_dir: str) -> tuple[list[str], list[str]]:
    """Return the engine's build command but the certificate authorities, in two parts.

    The mark and the TOOLCHAIN go on the image as labels (SOURCE_LABEL,
    TOOLCHAIN_LABEL) given on the command line. A docker build is given each proxy
    variable this process holds (config.proxy_name_list) by name, which it fills from
    its environment. A podman build copies them itself.

    Args:
        engine_path: The engine's executable.
        image_name: The name the image is given.
        toolchain: The TOOLCHAIN it is built with.
        mark: source_mark() of repo_dir.
        repo_dir: codetwine-loom's repository, the build's context.

    Returns:
        (what comes before the certificate authorities, what comes after them).
    """
    head_list = [engine_path, "build", "--build-arg", f"TOOLCHAIN={toolchain}",
                 "--label", f"{SOURCE_LABEL}={mark}",
                 "--label", f"{TOOLCHAIN_LABEL}={toolchain}"]
    if os.path.basename(engine_path) != "podman":
        for name in config.proxy_name_list():
            head_list += ["--build-arg", name]
    return head_list, ["-t", image_name, repo_dir]


def _cert_block_list(text: str) -> list[str]:
    """Return the certificates of a PEM text, each a block from BEGIN to END."""
    return _CERT_BLOCK_RE.findall(text)


def _cert_key(block: str) -> str:
    """Return a certificate block without its blank space, to compare blocks by."""
    return "".join(block.split())


def _mozilla_cert_key_set() -> set[str]:
    """Return _cert_key() of every certificate of Mozilla's bundle (certifi), if installed."""
    try:
        import certifi
        with open(certifi.where(), encoding="utf-8", errors="replace") as fp:
            return {_cert_key(block) for block in _cert_block_list(fp.read())}
    except (ImportError, OSError):
        return set()


def ca_pem_file() -> str:
    """Write the certificates CA_PEM passes, and return the file they are in.

    They are the certificates of the file config.CA_FILE_ENV names that Mozilla's bundle
    (certifi, the certificate authorities the image comes with) does not hold; anything
    else of the file, a private key among them, is left out. The file is
    <XDG_CACHE_HOME or ~/.cache>/codetwine-loom/ca-pem/<sha256 of them>.pem, readable by
    the user only.

    Returns:
        The path, or "" when config.CA_FILE_ENV names no file or leaves no certificate
        to pass.

    Raises:
        OSError: When the file of certificate authorities cannot be read, or the
            certificates cannot be written.
        ValueError: When the certificates to pass are longer than CA_PEM_MAX_BYTE.
    """
    ca_file = os.environ.get(config.CA_FILE_ENV, "")
    if not ca_file:
        return ""
    with open(ca_file, encoding="utf-8", errors="replace") as fp:
        block_list = _cert_block_list(fp.read())
    mozilla_key_set = _mozilla_cert_key_set()
    keep_list = [block for block in dict.fromkeys(block_list)
                 if _cert_key(block) not in mozilla_key_set]
    if not keep_list:
        return ""
    pem = "\n".join(keep_list) + "\n"
    pem_byte = len(pem.encode("utf-8"))
    if pem_byte > CA_PEM_MAX_BYTE:
        raise ValueError(
            f"{config.CA_FILE_ENV} ({ca_file}) holds {len(keep_list)} certificate(s) "
            f"Mozilla's bundle does not, {pem_byte} bytes, more than the "
            f"{CA_PEM_MAX_BYTE} an image build can be given. Point {config.CA_FILE_ENV} "
            f"at a file of only the certificate authorities you add")
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    pem_dir = os.path.join(cache_root, _CA_PEM_DIR)
    os.makedirs(pem_dir, mode=0o700, exist_ok=True)
    pem_file = os.path.join(pem_dir,
                            hashlib.sha256(pem.encode("utf-8")).hexdigest() + ".pem")
    if not os.path.isfile(pem_file):
        fd = os.open(pem_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(pem)
    return pem_file


def build_argv(engine_path: str, image_name: str, toolchain: str, mark: str,
               repo_dir: str) -> list[str]:
    """Return the command line that builds an image.

    The certificates of the file config.CA_FILE_ENV names that Mozilla's bundle does not
    hold (ca_pem_file) are passed as CA_PEM.

    Args:
        engine_path: The engine's executable.
        image_name: The name the image is given.
        toolchain: The TOOLCHAIN it is built with.
        mark: source_mark() of repo_dir.
        repo_dir: codetwine-loom's repository, the build's context.

    Returns:
        The command line.

    Raises:
        OSError: When the file of certificate authorities cannot be read, or the
            certificates cannot be written (ca_pem_file).
        ValueError: When the certificates to pass are longer than CA_PEM_MAX_BYTE.
    """
    head_list, tail_list = _build_part_tuple(engine_path, image_name, toolchain, mark,
                                             repo_dir)
    pem_file = ca_pem_file()
    ca_list = []
    if pem_file:
        with open(pem_file, encoding="utf-8") as fp:
            ca_list = ["--build-arg", f"{_CA_BUILD_ARG}={fp.read()}"]
    return [*head_list, *ca_list, *tail_list]


def build_command_text(engine_path: str, image_name: str, toolchain: str, mark: str,
                       repo_dir: str) -> str:
    """Return the command that builds an image, as it is typed in a shell.

    Examples:
        podman build --build-arg TOOLCHAIN=java
            --label io.codetwine-loom.source=<mark> --label io.codetwine-loom.toolchain=java
            --build-arg CA_PEM="$(cat /home/u/.cache/codetwine-loom/ca-pem/<hash>.pem)"
            -t codetwine-loom:java /opt/ctl

    Args:
        engine_path: The engine's executable.
        image_name: The name the image is given.
        toolchain: The TOOLCHAIN it is built with.
        mark: source_mark() of repo_dir.
        repo_dir: codetwine-loom's repository, the build's context.

    Returns:
        build_argv()'s command with the engine by its name and the certificates
        ca_pem_file() writes read from that file by the shell, on one line.

    Raises:
        OSError: As build_argv() raises it.
        ValueError: As build_argv() raises it.
    """
    head_list, tail_list = _build_part_tuple(os.path.basename(engine_path), image_name,
                                             toolchain, mark, repo_dir)
    pem_file = ca_pem_file()
    ca_text = f' --build-arg {_CA_BUILD_ARG}="$(cat {shlex.quote(pem_file)})"' \
        if pem_file else ""
    return f"{shlex.join(head_list)}{ca_text} {shlex.join(tail_list)}"

