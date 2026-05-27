"""
Generate golden test vectors for the mul_and RTL module straight from napl's
functional Python model (napl.operation.mul_and) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/mul_and.vec, one line per input combination:

    <in_0> <in_1> <out_unipolar> <out_bipolar>      (each 0/1, space-separated)

mul_and is combinational and stateless, so an exhaustive sweep of the two 1-bit
inputs (4 rows) fully characterizes it. The same generator pattern extends to
stateful ops by driving a multi-cycle stream and recording per-cycle I/O.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_mul_and.py
"""
import itertools
from pathlib import Path

import torch
from napl.operation import mul_and

VEC = Path(__file__).resolve().parent.parent / "vec" / "mul_and.vec"


def main():
    uni = mul_and(config={"polarity": "unipolar"})
    bi = mul_and(config={"polarity": "bipolar"})

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for a, b in itertools.product((0, 1), repeat=2):
            in_0, in_1 = torch.tensor(a), torch.tensor(b)
            out_uni = int(uni(in_0, in_1).item())
            out_bi = int(bi(in_0, in_1).item())
            f.write(f"{a} {b} {out_uni} {out_bi}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()
