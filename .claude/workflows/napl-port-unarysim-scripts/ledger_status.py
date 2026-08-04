#!/usr/bin/env python3
"""Parse the per-skill reports under reports/ and print a JSON status digest to stdout.

The napl-port-unarysim workflow uses this to decide what is already done WITHOUT having an
LLM eyeball markdown tables for exact-string matches. Done-state lives in the single-source
skill reports (the workflow's serialized recorders write there too); each is a markdown
table this parser extracts the key column(s) from deterministically.

Output (stdout) is a single JSON object:
    {
      "validated":    ["operation.mul_gaines", ...],   # napl-validate-unarysim-report.md  col "napl module"
      "rtl_verified": ["mul_gaines", ...],              # napl-gen-rtl-report.md  rows Status=verified
      "rtl_skipped":  ["wta", ...],                  # napl-gen-rtl-report.md  rows Status=skipped
      "rtl_failed":   [...],                          # napl-gen-rtl-report.md  rows Status=failed
      "improved":     {"src/napl/sim/operation/mul_gaines.py": "<git-hash>", ...}  # napl-opt-sim-report.md Source->Hash
    }

Missing report files yield empty collections (a fresh project has none).
"""
import argparse
import json
import os


def data_rows(path, header_marker):
    """Return the table body as a list of stripped cell-lists.

    A data row starts with '|', is not the header (identified by `header_marker`),
    and is not the '|---|' separator (which contains only pipe/dash/colon/space).
    """
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for ln in f.read().splitlines():
            s = ln.strip()
            if not s.startswith("|"):
                continue
            if header_marker in s:
                continue
            if not (set(s) - set("|-: ")):
                continue
            out.append([c.strip() for c in s.strip("|").split("|")])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="reports/napl-validate-unarysim-report.md")
    ap.add_argument("--rtl", default="reports/napl-gen-rtl-report.md")
    ap.add_argument("--improve", default="reports/napl-opt-sim-report.md")
    a = ap.parse_args()

    validated = [r[1] for r in data_rows(a.validation, "napl module") if len(r) > 1]

    rtl = {"verified": [], "skipped": [], "failed": []}
    for r in data_rows(a.rtl, "napl class"):
        if len(r) > 3:
            status = r[3].lower()
            if status in rtl:
                rtl[status].append(r[1])

    improved = {r[2]: r[3] for r in data_rows(a.improve, "napl kernel") if len(r) > 3 and r[2]}

    print(json.dumps({
        "validated": sorted(dict.fromkeys(validated)),
        "rtl_verified": sorted(dict.fromkeys(rtl["verified"])),
        "rtl_skipped": sorted(dict.fromkeys(rtl["skipped"])),
        "rtl_failed": sorted(dict.fromkeys(rtl["failed"])),
        "improved": improved,
    }, indent=2))


if __name__ == "__main__":
    main()
