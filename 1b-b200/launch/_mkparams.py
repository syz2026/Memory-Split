#!/usr/bin/env python3
"""Build an SSM AWS-RunShellScript parameters file.

Files are shipped base64-encoded so nothing in their contents has to survive
two layers of shell quoting. Usage:

    _mkparams.py OUT.json  --put LOCAL:REMOTE[:MODE] ...  --run 'shell command' ...
"""
import base64
import json
import sys


def main(argv):
    out = argv[0]
    commands = []
    i = 1
    while i < len(argv):
        flag = argv[i]
        if flag == "--put":
            spec = argv[i + 1].split(":")
            local, remote = spec[0], spec[1]
            mode = spec[2] if len(spec) > 2 else "644"
            with open(local, "rb") as fh:
                blob = base64.b64encode(fh.read()).decode()
            commands.append(f"echo '{blob}' | base64 -d > {remote}")
            commands.append(f"chmod {mode} {remote}")
            commands.append(f"echo 'wrote {remote}' && wc -c {remote}")
            i += 2
        elif flag == "--run":
            commands.append(argv[i + 1])
            i += 2
        else:
            sys.exit(f"unknown flag {flag}")

    with open(out, "w") as fh:
        json.dump({"commands": commands}, fh)
    print(f"{out}: {len(commands)} commands")


if __name__ == "__main__":
    main(sys.argv[1:])
