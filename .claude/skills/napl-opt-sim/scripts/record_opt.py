#!/usr/bin/env python3
"""Append an optimization record to reports/napl-opt-sim-report.md (creating it with a header if missing).

The napl-opt-sim skill calls this as its final step so every speedup run leaves a durable,
uniform row: which kernel was optimized, whether it was changed, the test-gate result, and the
before/after CPU and GPU (MPS) runtime with the measured speedup on identical inputs.

Example:
    python record_opt.py \
        --kernel operation.add_any --changed yes \
        --gate "PASS (add_any + downstream sweep 46/46)" \
        --cpu "51.6->52.0 ms (1.00x)" \
        --gpu "22.6->17.3 ms (1.30x)" \
        --summary "in-place accumulator add_/clamp_ once shape matches; removes 2 allocs/timestep"
"""
import argparse
import datetime
import os

HEADER = """# napl optimization (speedup) log

Records produced by the `napl-opt-sim` skill - one row per optimization run. Each row reports a
behavior-preserving speedup attempt for one kernel: whether code changed, the correctness gate
(the kernel's `test_<kernel>.py`, plus the downstream sweep when the kernel is a shared primitive),
and the before/after wall-clock on CPU and GPU (MPS) with the measured speedup. **Speedup** is the
kernel's own pre-optimization runtime divided by its optimized runtime on IDENTICAL inputs (GPU
timed with `torch.mps.synchronize()` before stopping the clock); runtimes are per-workload and not
comparable across rows. A `Changed=no` row (no safe speedup found, ~1.00x) is recorded too, so a
known-near-optimal kernel is not re-attempted blindly. **Source file** + **Hash** (the file's
`git hash-object` value after the run) double as the improve idempotency key: the napl-port-unarysim
workflow skips a file whose current hash matches a recorded row and re-optimizes one that differs.

| Date | napl kernel | Source file | Hash | Changed | Test gate | CPU base->opt (speedup) | GPU/MPS base->opt (speedup) | What changed |
|------|-------------|-------------|------|---------|-----------|-------------------------|-----------------------------|--------------|
"""


def cell(s):
    # pipes break markdown table columns; escape them
    return str(s).replace("|", "\\|").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reports/napl-opt-sim-report.md")
    ap.add_argument("--kernel", required=True, help="napl kernel, e.g. operation.add_any")
    ap.add_argument("--source", default="", help="repo-relative source file, e.g. src/napl/operation/add.py")
    ap.add_argument("--hash", default="", help="`git hash-object <source>` after the run (improve idempotency key)")
    ap.add_argument("--changed", required=True, help="yes or no")
    ap.add_argument("--gate", required=True, help='test-gate result, e.g. "PASS (add_any + sweep 46/46)"')
    ap.add_argument("--cpu", default="", help='CPU before->after (speedup), e.g. "51.6->52.0 ms (1.00x)"')
    ap.add_argument("--gpu", default="", help='GPU/MPS before->after (speedup), or "n/a"')
    ap.add_argument("--summary", default="", help="what changed + perf rationale, or why nothing changed")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    date = a.date or datetime.date.today().isoformat()
    row = "| " + " | ".join(cell(x) for x in (
        date, a.kernel, a.source, a.hash, a.changed, a.gate, a.cpu, a.gpu, a.summary)) + " |"

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
