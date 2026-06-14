#!/usr/bin/env python3
"""Append a validation record to reports/napl-validate-unarysim-report.md (creating it with a header if missing).

The napl-validate-unarysim skill calls this as its final step so every validation leaves a
durable, uniform row in the log: module, date, bit-exact result, agreement, and the napl
module's CPU/GPU runtime for the validated workload.

Example:
    python record_validation.py \
        --module module.linear_fsu --ref FSULinear \
        --bitexact "no (internal weight RNG)" \
        --agreement "RMSE 0.0014 vs SC bound 0.031" \
        --cpu "93.2 ms" --gpu "53.4 ms" \
        --regimes "bipolar, unipolar; B=64 in=512 out=256 T=256"
"""
import argparse
import datetime
import os

HEADER = """# napl vs UnarySim validation log

Records produced by the `napl-validate-unarysim` skill - one row per validation run. **Bit-exact**
means `torch.equal` held (max abs diff 0): deterministic ports (metrics, gate ops) should be exact,
while RNG-driven streaming kernels (`linear_fsu`, `conv_fsu`) only agree within the
stochastic-computing bound ~1/sqrt(N) and are recorded as not bit-exact with their agreement RMSE.
**Runtimes** are the napl module's wall-clock on CPU vs GPU (MPS) for the noted workload (GPU timed
with `torch.mps.synchronize()`); they are per-workload, not comparable across rows.

| Date | napl module | UnarySim ref | Bit-exact | Agreement | CPU runtime | GPU (MPS) runtime | Regimes / workload |
|------|-------------|--------------|-----------|-----------|-------------|-------------------|--------------------|
"""


def cell(s):
    # pipes break markdown table columns; escape them
    return str(s).replace("|", "\\|").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="reports/napl-validate-unarysim-report.md")
    ap.add_argument("--module", required=True, help="napl module, e.g. module.linear_fsu")
    ap.add_argument("--ref", required=True, help="UnarySim class, e.g. FSULinear")
    ap.add_argument("--bitexact", required=True, help='e.g. "yes (diff 0)" or "no (weight RNG)"')
    ap.add_argument("--agreement", default="exact", help="tolerance/RMSE vs SC bound when not bit-exact")
    ap.add_argument("--cpu", required=True, help='CPU runtime, e.g. "93.2 ms"')
    ap.add_argument("--gpu", required=True, help='GPU/MPS runtime, e.g. "53.4 ms" or "n/a"')
    ap.add_argument("--regimes", default="", help="polarities/known-answers/workload shape")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    a = ap.parse_args()

    date = a.date or datetime.date.today().isoformat()
    row = "| " + " | ".join(cell(x) for x in (
        date, a.module, a.ref, a.bitexact, a.agreement, a.cpu, a.gpu, a.regimes)) + " |"

    # collect existing data rows (the table body), if any
    new = not os.path.exists(a.file)
    rows = []
    if not new:
        with open(a.file) as f:
            lines = f.read().splitlines()
        # data rows are pipe-rows after the header/separator that are not the separator itself
        for ln in lines:
            s = ln.strip()
            if s.startswith("|") and "Date | napl module" not in s and set(s) - set("|-: "):
                rows.append(ln.rstrip())

    rows.append(row)
    # rank the log by module name (col 2) only, not by date
    def key(r):
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        return parts[1].lower()  # module name only
    rows = sorted(dict.fromkeys(rows), key=key)  # dedupe identical rows, then sort

    os.makedirs(os.path.dirname(a.file) or ".", exist_ok=True)
    with open(a.file, "w") as f:
        f.write(HEADER + "\n".join(rows) + "\n")
    print(("created " if new else "updated ") + a.file + " (sorted by module):")
    print(row)


if __name__ == "__main__":
    main()
