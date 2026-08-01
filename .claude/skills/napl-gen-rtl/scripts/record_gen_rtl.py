#!/usr/bin/env python3
"""Append an RTL-generation record to reports/napl-gen-rtl-report.md (creating it with a header if missing).

The napl-gen-rtl skill calls this as its final step so every generation run leaves a durable,
uniform row: which operation class got Verilog RTL, the module(s) emitted, whether `make test`
verified it against the Python model, the pipeline delay written back to self.hw.pp_delay, and
the polarity variants covered.

Example:
    python record_gen_rtl.py \
        --class shiftreg --rtl shiftreg \
        --status verified --make-test PASS \
        --pp-delay 4 --polarities "none (bare op)" \
        --notes "depth-4 circular FIFO; non-zero i%2 reset; gen + tb from the Python model"
"""
import argparse
import datetime
import os

HEADER = """# napl RTL generation log

Records produced by the `napl-gen-rtl` skill - one row per RTL-generation run. Each row attests
that Verilog RTL was emitted for an `operation` class under `src/napl/imp/operation/<op>/` and
verified against the napl Python model via golden-vector co-simulation (`make test` PASS only on a
full match). **Status** is `verified` (make test PASSed), `skipped` (no sound gate-level mapping, so
no RTL), or `failed` (generation/verification did not pass). **pp_delay** is the RTL input-to-output
latency in `i_clk` cycles written back to the class's `self.hw.pp_delay` (min across paths when they
differ; blank for skipped). Golden vectors always come from the Python model, never a hand truth
table.

| Date | napl class | RTL module(s) | Status | make test | pp_delay (cyc) | Polarities | Notes |
|------|------------|---------------|--------|-----------|----------------|------------|-------|
"""


def cell(s):
    # Escape pipes to preserve Markdown table columns.
    return str(s).replace("|", "\\|").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reports/napl-gen-rtl-report.md")
    ap.add_argument("--class", dest="cls", required=True, help="napl operation class, e.g. shiftreg")
    ap.add_argument("--rtl", default="", help="RTL module name(s) emitted, e.g. 'mul_and_unipolar, mul_and_bipolar'")
    ap.add_argument("--status", required=True, help="verified | skipped | failed")
    ap.add_argument("--make-test", dest="make_test", default="", help="PASS / FAIL / n/a")
    ap.add_argument("--pp-delay", dest="pp_delay", default="", help="pp_delay in cycles, or blank if skipped")
    ap.add_argument("--polarities", default="", help='e.g. "unipolar, bipolar" or "none (bare op)"')
    ap.add_argument("--notes", default="", help="structure, skip reason, or failure cause")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    date = a.date or datetime.date.today().isoformat()
    row = "| " + " | ".join(cell(x) for x in (
        date, a.cls, a.rtl, a.status, a.make_test, a.pp_delay, a.polarities, a.notes)) + " |"

    new = not os.path.exists(a.file)
    rows = []
    if not new:
        with open(a.file) as f:
            lines = f.read().splitlines()
        for ln in lines:
            s = ln.strip()
            if s.startswith("|") and "Date | napl class" not in s and set(s) - set("|-: "):
                rows.append(ln.rstrip())

    rows.append(row)
    # Sort by class name, independent of record date.
    def key(r):
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        return parts[1].lower()
    rows = sorted(dict.fromkeys(rows), key=key)

    os.makedirs(os.path.dirname(a.file) or ".", exist_ok=True)
    with open(a.file, "w") as f:
        f.write(HEADER + "\n".join(rows) + "\n")
    print(("created " if new else "updated ") + a.file + " (sorted by class):")
    print(row)


if __name__ == "__main__":
    main()
