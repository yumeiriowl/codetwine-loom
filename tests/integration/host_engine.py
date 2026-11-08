"""A container engine that runs the container's script on this machine.

It reads the command line sandbox.build_run_argv() builds for podman or docker, turns
the mounts into a path table, the -e arguments into the environment, and runs the
entry point script the image would run, with every path inside the container written
as the path on this machine. What the script writes lands where a container's would.

Usage:
    host_engine.py run [engine options] --entrypoint <script> <image> <tool args>
    host_engine.py ps --quiet --filter name=<container>
    host_engine.py rm --force <container>
    host_engine.py image inspect [--format {{.Id}}] <image>
    host_engine.py image inspect --format {{json .Config.Labels}} <image>
    host_engine.py build [--build-arg NAME[=VALUE]]... -t <image> <context>

The environment variable CTLOOM_HOST_ENGINE_LOG names a file every run is appended to,
as one JSON line holding the argv, the environment and the exit status.
CTLOOM_HOST_ENGINE_WORK names a directory the clone and the run's temporary files are put
under, in place of the directories the engine was told to mount at /work and /tmp.

The images: without CTLOOM_HOST_ENGINE_IMAGE_FILE every image is there, with the labels
{"io.codetwine-loom.source": CTLOOM_HOST_ENGINE_SOURCE}. With it, the file holds
{"image_dict": {image: labels}, "build_list": [the argv of each build]}: an image not in
image_dict is not there, and a build puts the image in with the labels its --label
arguments give, and appends its argv to build_list. A file that is not there holds
neither.
"""

import json
import os
import subprocess
import sys

# The scripts the image installs, by the name the engine is told to run
_ENTRYPOINT_DICT = {
    "/usr/local/bin/ctloom-sandbox": "entrypoint.sh",
    "/usr/local/bin/ctloom-warm": "warm.sh",
}

# The address scheme the broker's socket is named with inside a value
_UNIX_SCHEME = "unix://"

# The mounts that are moved under CTLOOM_HOST_ENGINE_WORK when it is set
_MOVE_INSIDE_TUPLE = ("/work", "/tmp")

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_run_argv(arg_list: list[str]) -> dict:
    """Read the engine's run arguments into the parts this engine needs.

    Args:
        arg_list: Everything after "run".

    Returns:
        {"mount_dict": {inside: host}, "env_dict": {name: value}, "env_name_list":
        [names given without a value], "entrypoint": the script's name inside the
        image, "image": the image, "arg_list": the tool's arguments}.
    """
    mount_dict, env_dict, env_name_list = {}, {}, []
    entrypoint, image, tool_arg_list = "", "", []
    i = 0
    while i < len(arg_list):
        arg = arg_list[i]
        if arg == "-v":
            host, inside = arg_list[i + 1].split(":")[:2]
            mount_dict[inside] = host
            i += 2
        elif arg == "-e":
            name, is_value, value = arg_list[i + 1].partition("=")
            if is_value:
                env_dict[name] = value
            else:
                env_name_list.append(name)
            i += 2
        elif arg == "--entrypoint":
            entrypoint = arg_list[i + 1]
            i += 2
        elif arg in ("--pids-limit", "--memory", "--name", "--user"):
            i += 2
        elif arg.startswith("-"):
            i += 1
        else:
            image = arg
            tool_arg_list = arg_list[i + 1:]
            break
    return {"mount_dict": mount_dict, "env_dict": env_dict,
            "env_name_list": env_name_list, "entrypoint": entrypoint, "image": image,
            "arg_list": tool_arg_list}


def to_host(value: str, mount_dict: dict[str, str]) -> str:
    """Write a value holding a path inside the container as the path on this machine.

    Examples:
        "/out/calc/x" with {"/out": "/tmp/o"}         -> "/tmp/o/calc/x"
        "unix:///run/ctloom/llm.sock" with {"/run/ctloom": "/tmp/s"} -> "unix:///tmp/s/llm.sock"
        "/output" with {"/out": "/tmp/o"}             -> "/output"

    Args:
        value: The value as the container would see it.
        mount_dict: {inside: host}, as parse_run_argv() returns it.

    Returns:
        The value with the longest matching mount replaced, or as it was.
    """
    prefix = _UNIX_SCHEME if value.startswith(_UNIX_SCHEME) else ""
    path = value[len(prefix):]
    for inside in sorted(mount_dict, key=len, reverse=True):
        if path == inside or path.startswith(inside + "/"):
            return prefix + mount_dict[inside] + path[len(inside):]
    return value


def child_env_dict(parsed_dict: dict, bin_dir: str) -> dict[str, str]:
    """Build the environment the container's script runs with.

    Only what the engine was given reaches it: the values passed with -e, the names
    passed bare where this process has them, a PATH holding the interpreter, and
    LITELLM_LOCAL_MODEL_COST_MAP and LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS as the image
    sets them.

    Args:
        parsed_dict: parse_run_argv()'s return value.
        bin_dir: The directory holding the "python" the script starts.

    Returns:
        {name: value}, with every path written as this machine's.
    """
    mount_dict = parsed_dict["mount_dict"]
    env_dict = {"PATH": f"{bin_dir}:/usr/bin:/bin", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "True"}
    for name in parsed_dict["env_name_list"]:
        if name in os.environ:
            env_dict[name] = os.environ[name]
    for name, value in parsed_dict["env_dict"].items():
        env_dict[name] = to_host(value, mount_dict)
    return env_dict


def _bin_dir(root: str) -> str:
    """Make the directory whose "python" starts this interpreter, and return it."""
    bin_dir = os.path.join(root, "host-engine-bin")
    os.makedirs(bin_dir, exist_ok=True)
    wrapper = os.path.join(bin_dir, "python")
    if not os.path.exists(wrapper):
        with open(wrapper, "w", encoding="utf-8") as fp:
            fp.write(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
        os.chmod(wrapper, 0o755)
    return bin_dir


def _moved_mount_dict(mount_dict: dict[str, str]) -> dict[str, str]:
    """Put the clone's and the temporary files' mounts under CTLOOM_HOST_ENGINE_WORK.

    The directories the engine is told to mount there sit under the output directory,
    which a container keeps apart from the clone by mounting them separately. Without
    the move the clone would be read as part of the output directory.

    Args:
        mount_dict: {inside: host}, as parse_run_argv() returns it.

    Returns:
        The same table, with the two mounts moved when the variable is set.
    """
    work_root = os.environ.get("CTLOOM_HOST_ENGINE_WORK")
    if not work_root:
        return mount_dict
    moved_dict = dict(mount_dict)
    for inside in _MOVE_INSIDE_TUPLE:
        if inside in mount_dict:
            scratch_name = os.path.basename(os.path.dirname(mount_dict[inside]))
            moved_dict[inside] = os.path.join(work_root, scratch_name, inside.strip("/"))
            os.makedirs(moved_dict[inside], exist_ok=True)
    return moved_dict


def _record(entry_dict: dict) -> None:
    """Append one run to the log CTLOOM_HOST_ENGINE_LOG names, when it names one."""
    path = os.environ.get("CTLOOM_HOST_ENGINE_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry_dict, ensure_ascii=False) + "\n")


def run(arg_list: list[str]) -> int:
    """Run one container's script here and return its exit status.

    Args:
        arg_list: Everything after "run".

    Returns:
        The script's exit status.
    """
    parsed_dict = parse_run_argv(arg_list)
    script = _ENTRYPOINT_DICT.get(parsed_dict["entrypoint"])
    if script is None:
        print(f"host_engine: unknown entrypoint {parsed_dict['entrypoint']}",
              file=sys.stderr)
        return 125
    mount_dict = parsed_dict["mount_dict"] = _moved_mount_dict(parsed_dict["mount_dict"])
    scratch_root = os.path.dirname(mount_dict.get("/tmp", os.getcwd()))
    env_dict = child_env_dict(parsed_dict, _bin_dir(scratch_root))
    tool_arg_list = [to_host(arg, mount_dict) for arg in parsed_dict["arg_list"]]
    proc = subprocess.run(["sh", os.path.join(_REPO_DIR, "sandbox", script),
                           *tool_arg_list], env=env_dict, check=False)
    _record({"argv": arg_list, "env": env_dict, "tool_args": tool_arg_list,
             "exit_status": proc.returncode})
    return proc.returncode


# The format of the image inspect that asks for an image's labels
_LABEL_FORMAT = "{{json .Config.Labels}}"


def _image_record() -> dict | None:
    """Read the file CTLOOM_HOST_ENGINE_IMAGE_FILE names, or None when it names none."""
    path = os.environ.get("CTLOOM_HOST_ENGINE_IMAGE_FILE")
    if not path:
        return None
    if not os.path.exists(path):
        return {"image_dict": {}, "build_list": []}
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def inspect_label(image: str) -> int:
    """Print an image's labels as JSON and return 0, or 125 when it is not there."""
    record = _image_record()
    if record is None:
        print(json.dumps({"io.codetwine-loom.source":
                          os.environ.get("CTLOOM_HOST_ENGINE_SOURCE", "")}))
        return 0
    if image not in record["image_dict"]:
        print(f"Error: {image}: image not known", file=sys.stderr)
        return 125
    print(json.dumps(record["image_dict"][image]))
    return 0


def build(arg_list: list[str]) -> int:
    """Put the image a build names into CTLOOM_HOST_ENGINE_IMAGE_FILE and return 0.

    Args:
        arg_list: Everything after "build".
    """
    record = _image_record()
    if record is None:
        return 0
    label_dict, image = {}, ""
    for index, arg in enumerate(arg_list):
        if arg == "--label":
            name, _, value = arg_list[index + 1].partition("=")
            label_dict[name] = value
        elif arg == "-t":
            image = arg_list[index + 1]
    record["image_dict"][image] = label_dict
    record["build_list"].append(arg_list)
    with open(os.environ["CTLOOM_HOST_ENGINE_IMAGE_FILE"], "w", encoding="utf-8") as fp:
        json.dump(record, fp)
    return 0


def main(argv: list[str]) -> int:
    """Answer one engine command.

    Args:
        argv: The arguments after the program's name.

    Returns:
        The exit status.
    """
    if not argv:
        return 2
    if argv[0] == "run":
        return run(argv[1:])
    if argv[:4] == ["image", "inspect", "--format", "{{.Id}}"]:
        print(f"sha256:host-engine-{argv[-1]}")
        return 0
    if argv[:4] == ["image", "inspect", "--format", _LABEL_FORMAT]:
        return inspect_label(argv[-1])
    if argv[0] == "build":
        return build(argv[1:])
    if argv[0] in ("ps", "rm", "image"):
        return 0
    print(f"host_engine: unknown command {argv[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
