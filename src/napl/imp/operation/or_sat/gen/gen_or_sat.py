"""
Generate golden test vectors for the or_sat RTL module straight from napl's
functional Python model (napl.sim.operation.or_sat) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

or_sat is a combinational, stateless, unipolar-only saturating add (a bitwise OR
that realizes a + b - a*b on two independent rate-coded streams). It holds no
state and has no reset, so the vectors are a plain per-cycle spike list with no
reset rows.

The two operands must be independent, so each is encoded on its own Sobol
dimension, mirroring test_or_sat.py's encoder_dims [1, 2] and its make_values
range (the full unipolar span, a = linspace(0,1), b = linspace(1,0)).

Output: ../vec/or_sat.vec, one line per cycle:

    <in_input_0> <in_input_1> <out>      (each 0/1, space-separated)

or_sat has no sizing config, so there is a single bare module or_sat and no
parameter header.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_or_sat.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import or_sat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams

VEC = Path(__file__).resolve().parent.parent / "vec" / "or_sat.vec"

# Encoder settings mirror test_or_sat.py: full-range unipolar streams on distinct
# Sobol dimensions (encoder_dims [1, 2]) so the two operands are decorrelated.
CODEC0 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def main():
    model = or_sat({"polarity": "unipolar"})

    # test_or_sat.py make_values: a sweeps up, b sweeps down, over the full
    # unipolar legal range [0, 1] with no narrowing.
    a = torch.linspace(0.0, 1.0, 128)
    b = torch.linspace(1.0, 0.0, 128)
    pairs = list(zip(a.tolist(), b.tolist()))

    s0, s1 = pair_streams(CODEC0, CODEC1, pairs)

    rows = []  # (in_input_0, in_input_1, out)
    model.reset()
    for x, y in zip(s0, s1):
        out = int(model(torch.tensor(x), torch.tensor(y)).item())
        rows.append((x, y, out))

    VEC.parent.mkdir(parents=True, exist_ok=True)
    with VEC.open("w") as f:
        for x, y, out in rows:
            f.write(f"{x} {y} {out}\n")
    print(f"wrote {VEC} ({len(rows)} vectors)")


if __name__ == "__main__":
    main()
