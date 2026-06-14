#!/usr/bin/env python3
"""Append a sim-vs-RTL validation record to reports/napl-validate-sim-rtl-report.md (creating it with a header if missing).

The napl-validate-sim-rtl skill calls this as its final step so every validation leaves a
durable, uniform row: which kernel's Python model was checked against which RTL module, on
the test-derived inputs, with the bit-exact result and the regimes/reset events covered.

Example:
    python record_sim_rtl.py \
        --kernel operation.shiftreg --rtl shiftreg \
        --result PASS --vectors "2048/2048" \
        --polarities bipolar \
        --reset "power-on + mid-stream" \
        --regimes "depth=4 T=256 sobol; corners {-1,0,+1} + 4 distribution draws" \
        --notes "non-zero i%2 reset; mid-stream reset from dirtied FIFO matched"
"""
import argparse
import datetime
import os

HEADER = """# napl sim-vs-RTL validation log

Records produced by the `napl-validate-sim-rtl` skill - one row per validation run. Each row
attests that the napl Python functional model and its Verilog RTL produced **bit-exact identical**
outputs (cycle-for-cycle, `make test` PASS only on a full match) on the streams the kernel's
`test_<kernel>.py` encodes, fed to both sides. **Result** is PASS (every vector matched) or FAIL
(with the diverging cycle / cause in Notes). **Reset** records which reset cases were exercised:
power-on (`i_rst_n` at t=0 vs `reset()`) and, for stateful ops, a mid-stream reset from a dirtied
state (the check a power-on-only run misses). **Vectors** is matched/total. Bit-exact is the only
acceptable PASS: spikes are 0/1 and the RTL is a faithful gate-level model, so outputs are identical
or there is a real bug, never "close".

| Date | napl kernel | RTL module | Result | Vectors | Polarities | Reset checks | Regimes / streams | Notes |
|------|-------------|------------|--------|---------|------------|--------------|-------------------|-------|
"""


def cell(s):
    # pipes break markdown table columns; escape them
    return str(s).replace("|", "\\|").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reports/napl-validate-sim-rtl-report.md")
    ap.add_argument("--kernel", required=True, help="napl kernel, e.g. operation.shiftreg")
    ap.add_argument("--rtl", required=True, help="RTL module / OP name, e.g. shiftreg")
    ap.add_argument("--result", required=True, help="PASS or FAIL")
    ap.add_argument("--vectors", default="", help='matched/total, e.g. "2048/2048"')
    ap.add_argument("--polarities", default="", help='e.g. "unipolar, bipolar" or "bipolar"')
    ap.add_argument("--reset", default="", help='reset cases exercised, e.g. "power-on + mid-stream"')
    ap.add_argument("--regimes", default="", help="test-derived streams: corners, draws, depth, timestep")
    ap.add_argument("--notes", default="", help="divergence cause if FAIL, or noteworthy detail")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    date = a.date or datetime.date.today().isoformat()
    row = "| " + " | ".join(cell(x) for x in (
        date, a.kernel, a.rtl, a.result, a.vectors, a.polarities, a.reset, a.regimes, a.notes)) + " |"

    # collect existing data rows (the table body), if any
    new = not os.path.exists(a.file)
    rows = []
    if not new:
        with open(a.file) as f:
            lines = f.read().splitlines()
        for ln in lines:
            s = ln.strip()
            if s.startswith("|") and "Date | napl kernel" not in s and set(s) - set("|-: "):
                rows.append(ln.rstrip())

    rows.append(row)
    # rank the log by kernel name (col 2) only, not by date
    def key(r):
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        return parts[1].lower()  # kernel name only
    rows = sorted(dict.fromkeys(rows), key=key)  # dedupe identical rows, then sort

    os.makedirs(os.path.dirname(a.file) or ".", exist_ok=True)
    with open(a.file, "w") as f:
        f.write(HEADER + "\n".join(rows) + "\n")
    print(("created " if new else "updated ") + a.file + " (sorted by kernel):")
    print(row)


if __name__ == "__main__":
    main()
