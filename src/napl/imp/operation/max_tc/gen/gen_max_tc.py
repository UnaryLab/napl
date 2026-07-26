"""
Generate golden test vectors for the max_tc RTL module straight from napl's
functional Python model (napl.sim.operation.max_tc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/max_tc.vec, one line per input combination:

    <in_0> <in_1> <out>      (each 0/1, space-separated)

max_tc is combinational and stateless (OR of two temporal-coded spikes), so an
exhaustive sweep of the two 1-bit inputs (4 rows) fully characterizes it.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_max_tc.py
"""
import itertools
from pathlib import Path

import torch
from napl.sim.operation import max_tc

VEC = Path(__file__).resolve().parent.parent / "vec" / "max_tc.vec"


def main():
    model = max_tc(config={})
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for a, b in itertools.product((0, 1), repeat=2):
            in_0, in_1 = torch.tensor(a), torch.tensor(b)
            out = int(model(in_0, in_1).item())
            f.write(f"{a} {b} {out}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()
