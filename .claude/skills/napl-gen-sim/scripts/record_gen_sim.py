#!/usr/bin/env python3
"""Append a port (UnarySim -> napl) record to reports/napl-gen-sim-report.md (creating it with a header if missing).

The napl-gen-sim skill calls this as its final step so every porting run leaves a durable, uniform
row: which UnarySim class was ported into which napl class, where it landed, the test created for it,
and whether the port was validated faithful against UnarySim.

Example:
    python record_gen_sim.py \
        --napl module.linear_fsu_pc --unarysim FSULinearPC \
        --status ported --test tests/module/test_linear_fsu_pc.py \
        --validated "bit-exact vs FSULinearPC (CPU+MPS)" \
        --notes "parallel-counter accumulation variant of FSULinear; streaming FSU kernel"
"""
import argparse
import datetime
import os

HEADER = """# napl port-from-UnarySim (gen-sim) log

Records produced by the `napl-gen-sim` skill - one row per UnarySim class ported into napl. Each row
attests that a UnarySim implementation was reimplemented in napl's conventions (napl_base subclass,
config/key_list validation, stype/ntype dtypes, the streaming `@napl_sim_timesteps` paradigm or the
single-shot binary-domain form, the hw_params contract, napl naming), given a `test_<name>.py`, and
validated faithful against the UnarySim original. **Status** is `ported` (new napl class created and
validated), `skipped` (no sound napl mapping, e.g. a backward-only autograd helper), or `failed`.
**Validated** records the napl-vs-UnarySim agreement (bit-exact or within the SC/quant bound).

| Date | napl class | UnarySim source | Status | Test | Validated | Notes |
|------|------------|-----------------|--------|------|-----------|-------|
"""


def cell(s):
    # pipes break markdown table columns; escape them
    return str(s).replace("|", "\\|").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reports/napl-gen-sim-report.md")
    ap.add_argument("--napl", required=True, help="napl target class, exactly <subpackage>.<name>, e.g. module.linear_fsu_pc")
    ap.add_argument("--unarysim", required=True, help="UnarySim source class, e.g. FSULinearPC")
    ap.add_argument("--status", required=True, help="ported | skipped | failed")
    ap.add_argument("--test", default="", help="test file created, e.g. tests/module/test_linear_fsu_pc.py")
    ap.add_argument("--validated", default="", help="napl-vs-UnarySim agreement (bit-exact / RMSE vs SC bound), or why not")
    ap.add_argument("--notes", default="", help="paradigm, subpackage, skip reason, or failure cause")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    date = a.date or datetime.date.today().isoformat()
    row = "| " + " | ".join(cell(x) for x in (
        date, a.napl, a.unarysim, a.status, a.test, a.validated, a.notes)) + " |"

    # collect existing data rows (the table body), if any
    new = not os.path.exists(a.file)
    rows = []
    if not new:
        with open(a.file) as f:
            lines = f.read().splitlines()
        for ln in lines:
            s = ln.strip()
            if s.startswith("|") and "napl class | UnarySim source" not in s and set(s) - set("|-: "):
                rows.append(ln.rstrip())

    rows.append(row)
    # rank the log by napl class name (col 2) only, not by date
    def key(r):
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        return parts[1].lower()  # napl class name only
    rows = sorted(dict.fromkeys(rows), key=key)  # dedupe identical rows, then sort

    os.makedirs(os.path.dirname(a.file) or ".", exist_ok=True)
    with open(a.file, "w") as f:
        f.write(HEADER + "\n".join(rows) + "\n")
    print(("created " if new else "updated ") + a.file + " (sorted by napl class):")
    print(row)


if __name__ == "__main__":
    main()
