#!/usr/bin/env python3
"""Collect bounded, read-only facts from an unknown MIT Slurm cluster."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


COMMANDS = (
    ("sinfo", "--version"),
    ("sinfo", "-h", "-o", "%P|%G|%l|%D|%m"),
    ("scontrol", "show", "config"),
)
COMMAND_NAMES = ("slurm_version", "partitions", "config")
COMMAND_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 1_048_576
SCRATCH_ENVIRONMENT = ("SCRATCH", "SLURM_TMPDIR", "TMPDIR", "HOME")
_CAPACITY_RE = re.compile(r"^(?P<value>[0-9]+)(?P<suffix>[+*]?)$")


def _bounded(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)[:MAX_OUTPUT_CHARS]


def _run(command: Sequence[str], runner: Callable) -> dict:
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        returncode = int(completed.returncode)
        stdout, stderr = _bounded(completed.stdout), _bounded(completed.stderr)
        result = {
            "argv": list(command),
            "ok": returncode == 0,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        if returncode:
            result["error"] = stderr.strip() or stdout.strip() or f"exit {returncode}"
        return result
    except Exception as error:
        return {
            "argv": list(command),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"{type(error).__name__}: {error}",
        }


def _capacity(raw: str, *, field: str, line: int) -> tuple[int, str | None]:
    match = _CAPACITY_RE.fullmatch(raw)
    if match is None:
        raise ValueError(f"partition row {line} {field} capacity {raw!r} is malformed")
    return int(match.group("value")), match.group("suffix") or None


def parse_partitions(text: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split("|")]
        if len(fields) != 5:
            raise ValueError(f"partition row {line_number} must have five fields")
        raw_name, gres, limit, raw_nodes, raw_memory = fields
        default = raw_name.endswith("*")
        name = raw_name[:-1] if default else raw_name
        nodes, node_suffix = _capacity(raw_nodes, field="node", line=line_number)
        memory, memory_suffix = _capacity(
            raw_memory,
            field="memory",
            line=line_number,
        )
        rows.append(
            {
                "partition": name,
                "default": default,
                "gres": gres,
                "time_limit": limit,
                "nodes": nodes,
                "nodes_approximate": node_suffix is not None,
                "nodes_suffix": node_suffix,
                "memory_mb": memory,
                "memory_mb_approximate": memory_suffix is not None,
                "memory_mb_suffix": memory_suffix,
            }
        )
    return rows


def parse_slurm_config(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() and key.strip() not in values:
                values[key.strip()] = value.strip()
    return values


def collect_probe(
    *,
    command_runner: Callable = subprocess.run,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    disk_usage: Callable = shutil.disk_usage,
    path_exists: Callable[[str], bool] = os.path.isdir,
    source_root: Path | str = ".",
) -> dict:
    environment = os.environ if environ is None else environ
    commands = {
        name: _run(command, command_runner)
        for name, command in zip(COMMAND_NAMES, COMMANDS)
    }
    partitions = []
    partition_error = None
    if commands["partitions"]["ok"]:
        try:
            partitions = parse_partitions(commands["partitions"]["stdout"])
        except ValueError as error:
            partition_error = f"{type(error).__name__}: {error}"
            commands["partitions"]["ok"] = False
            commands["partitions"]["error"] = partition_error
    scratch = []
    for name in SCRATCH_ENVIRONMENT:
        path = environment.get(name)
        if not path or not path_exists(path):
            continue
        try:
            usage = disk_usage(path)
            scratch.append(
                {
                    "environment": name,
                    "path": path,
                    "free_bytes": int(usage.free),
                    "total_bytes": int(usage.total),
                }
            )
        except Exception as error:
            scratch.append(
                {
                    "environment": name,
                    "path": path,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "schema_version": 1,
        "commands": commands,
        "slurm": {
            "version": (
                commands["slurm_version"]["stdout"].strip()
                if commands["slurm_version"]["ok"]
                else None
            ),
            "partitions": partitions,
            "config": parse_slurm_config(commands["config"]["stdout"]),
            "partition_parse_error": partition_error,
        },
        "module": {
            "available": bool(
                which("module")
                or environment.get("MODULESHOME")
                or environment.get("LMOD_CMD")
            )
        },
        "python_executables": [
            candidate
            for candidate in (sys.executable, which("python3"), which("python"))
            if candidate
        ],
        "scratch_candidates": scratch,
        "source_root": str(Path(source_root)),
    }


def write_probe(path: Path | str, evidence: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record bounded read-only Slurm, filesystem, module, and Python facts.",
        epilog="A trailing '+' or '*' capacity is retained and marked approximate.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", default=".")
    args = parser.parse_args(argv)
    evidence = collect_probe(source_root=args.source_root)
    print(write_probe(args.out, evidence))
    return 0 if all(item["ok"] for item in evidence["commands"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
