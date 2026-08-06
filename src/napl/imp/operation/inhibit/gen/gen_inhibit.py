"""
Generate golden test vectors for the inhibit RTL module straight from napl's
functional Python model (napl.sim.operation.inhibit) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

inhibit is stateful: it holds a sticky 1-bit inhibition latch (reset to 0) that,
once set, forces the output low for the rest of the stream. Each encoded operand
pair is therefore one independent temporal-code segment preceded by a reset row,
mirroring the per-element latch the tensor model keeps in test_inhibit.py. Those
resets also exercise recovery from a dirtied latch.

Output: ../vec/inhibit.vec, one line per cycle:

    <rst_n> <in_data> <in_inhibit> <out>      (each 0/1, space-separated)

A line with rst_n==0 is a reset pulse: in_data/in_inhibit/out are don't-care (0) and the
TB asserts i_rst_n low for that cycle instead of checking the output. The RTL latch is
level-sensitive and clockless, so i_rst_n low clears it directly.

inhibit has no polarity branch (polarity_required=False) and no sizing config, so
there is a single bare module inhibit and no parameter header.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_inhibit.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import inhibit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "inhibit.vec"

# Encoder settings mirror test_inhibit.py and use distinct temporal dimensions.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "temporal", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "temporal", "dim": 2}


def main():
    model = inhibit(config={})

    rows = []  # (rst_n, in_data, in_inhibit, out)
    for pair in rep_pairs("bipolar", "bipolar"):
        model.reset()
        rows.append((0, 0, 0, 0))
        s0, s1 = pair_streams(CODEC0, CODEC1, [pair])
        for a, b in zip(s0, s1):
            out = int(model(torch.tensor(a), torch.tensor(b)).item())
            rows.append((1, a, b, out))

    VEC.parent.mkdir(parents=True, exist_ok=True)
    with VEC.open("w") as f:
        for rst_n, a, b, out in rows:
            f.write(f"{rst_n} {a} {b} {out}\n")
    print(f"wrote {VEC} ({len(rows)} vectors)")


if __name__ == "__main__":
    main()
