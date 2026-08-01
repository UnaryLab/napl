#!/usr/bin/env python3
"""Fetch a source file from github.com/diwu1990/UnarySim and save it locally.

Tries the authenticated GitHub API first (`gh api`, base64-encoded contents), then falls
back to raw.githubusercontent.com over plain HTTP. Normalizes CRLF line endings (UnarySim
files are often Windows-terminated) and prints the class names in the file so you can pick
the right one for the comparison harness.

Usage:
    python fetch_unarysim.py metric/metric.py
    python fetch_unarysim.py kernel/mul.py --out /tmp/us_mul.py
    python fetch_unarysim.py kernel/linear.py --show FSULinear   # also print that class body
"""
import argparse
import base64
import re
import subprocess
import sys
import urllib.request

REPO = "diwu1990/UnarySim"


def fetch_via_gh(path):
    try:
        out = subprocess.run(
            ["gh", "api", f"repos/{REPO}/contents/{path}", "--jq", ".content"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return base64.b64decode(out.stdout).decode("utf-8", "replace")
    except Exception:
        return None


def fetch_via_raw(path):
    for branch in ("master", "main"):
        url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def class_block(src, name):
    """Return the source of `class name` up to the next top-level class/def."""
    lines = src.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(rf"class {name}\b", l)), None)
    if start is None:
        return None
    out = [lines[start]]
    for l in lines[start + 1:]:
        if re.match(r"(class |def )", l):  # Stop at the next top-level definition.
            break
        out.append(l)
    return "\n".join(out).rstrip()


def main():
    ap = argparse.ArgumentParser(description="Fetch a UnarySim source file.")
    ap.add_argument("path", help="repo-relative path, e.g. metric/metric.py or kernel/mul.py")
    ap.add_argument("--out", default=None, help="output path (default /tmp/unarysim_<path>)")
    ap.add_argument("--show", default=None, help="also print the body of this class")
    args = ap.parse_args()

    src = fetch_via_gh(args.path) or fetch_via_raw(args.path)
    if src is None:
        sys.exit(f"error: could not fetch '{args.path}' from {REPO} "
                 f"(check the path; for the API path try `gh auth login`)")

    src = src.replace("\r\n", "\n").replace("\r", "\n")
    out = args.out or "/tmp/unarysim_" + args.path.replace("/", "_")
    with open(out, "w") as f:
        f.write(src)

    classes = re.findall(r"^class (\w+)", src, re.M)
    print(f"saved {args.path} -> {out}  ({len(src.splitlines())} lines)")
    print("classes:", ", ".join(classes) if classes else "(none found)")

    if args.show:
        block = class_block(src, args.show)
        if block is None:
            print(f"\n(class {args.show!r} not found in this file)")
        else:
            print(f"\n----- class {args.show} -----\n{block}")


if __name__ == "__main__":
    main()
