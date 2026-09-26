"""
Generate golden test vectors for the inhibit_tc RTL module straight from napl's
functional Python model (napl.sim.operation.inhibit_tc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

inhibit_tc is stateful: it holds a sticky 1-bit inhibition latch (reset to 0) that,
once set, forces the output low for the rest of the stream. Each encoded operand
pair is therefore one independent temporal-code segment preceded by a reset row,
mirroring the per-element latch the tensor model keeps in test_inhibit_tc.py. Those
resets also exercise recovery from a dirtied latch.

Output: ../vec/inhibit_tc.vec, one line per cycle:

    <rst_n> <in_data> <in_inhibit> <out>      (each 0/1, space-separated)

A line with rst_n==0 is a reset pulse: in_data/in_inhibit/out are don't-care (0) and the
TB asserts i_rst_n low for that cycle instead of checking the output. The RTL state is a
posedge i_clk flop with an asynchronous active-low reset, so i_rst_n low clears it directly.

inhibit_tc has no polarity branch (polarity_required=False) and no sizing config, so
there is a single bare module inhibit_tc and no parameter header.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_inhibit_tc.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import inhibit_tc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "inhibit_tc.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "inhibit_tc_params.vh"

# Encoder settings mirror test_inhibit_tc.py and use distinct temporal dimensions.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "temporal", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "temporal", "dim": 2}


def main():
    model = inhibit_tc(config={})

    rows = []  # (rst_n, in_data, in_inhibit, out)
    # The operand range is test_inhibit_tc.py's make_values range.
    for pair in rep_pairs("bipolar", "bipolar", range0=(-1.0, 1.0), range1=(-1.0, 1.0)):
        model.reset()
        rows.append((0, 0, 0, 0))
        s0, s1 = pair_streams(CODEC0, CODEC1, [pair])
        for a, b in zip(s0, s1):
            out = int(model(torch.tensor(a), torch.tensor(b)).item())
            rows.append((1, a, b, out))

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # The testbench counts only the driven rows, not the reset-pulse rows.
    compared = 0
    with VEC.open("w") as f:
        for rst_n, a, b, out in rows:
            f.write(f"{rst_n} {a} {b} {out}\n")
            if rst_n:
                compared += 1
    PARAMS.write_text(f"`define GEN_VECTORS {compared}\n")
    print(f"wrote {VEC} ({len(rows)} vectors)")


if __name__ == "__main__":
    main()
