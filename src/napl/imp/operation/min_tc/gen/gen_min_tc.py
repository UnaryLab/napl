#!/usr/bin/env python
"""Emit golden vectors for min_tc from the napl Python model.

min_tc is a stateless combinational op (temporal-coded AND). We drive the model
per timestep over all 4 input combinations and record (in_0, in_1, out), where
out is whatever the napl forward() emits. The RTL is checked against this column,
never a hand-derived truth table.
"""
import os
import torch

from napl.sim.operation import min_tc

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vec")
OUT_PATH = os.path.join(OUT_DIR, "min_tc.vec")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = min_tc({})
    model.reset()

    lines = []
    lines.append("in_0 in_1 out")
    for a in (0, 1):
        for b in (0, 1):
            in_0 = torch.tensor([a], dtype=torch.int8)
            in_1 = torch.tensor([b], dtype=torch.int8)
            out = model(in_0, in_1)
            o = int(out.reshape(-1)[0].item())
            lines.append(f"{a} {b} {o}")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {OUT_PATH} ({len(lines) - 1} vectors)")


if __name__ == "__main__":
    main()
