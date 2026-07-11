"""
Generate golden test vectors for the gt_rc RTL module straight from napl's
functional Python model (napl.operation.gt_rc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

gt_rc is stateful: it holds a 1-bit result register (dff, reset to 1) and a
2-bit skew counter inside sync_skewed (width=2, reset to 0). So a single stream
of (in_0, in_1) pairs is driven from model.reset() and per-cycle I/O recorded.
Because the op is stateful, the stream also exercises a MID-STREAM reset: after
the state is dirtied by the first batch of pairs, model.reset() is called and a
reset marker is emitted, then the stream continues -- proving the RTL's
active-low i_rst_n returns dff->1 / cnt->0 from a dirtied state, matching the
Python reset() exactly.

Output: ../vec/gt_rc.vec, one line per cycle:

    <rst_n> <in_0> <in_1> <out>      (each 0/1, space-separated)

A line with rst_n==0 is a reset pulse: in_0/in_1/out are don't-care (0) and the
TB asserts i_rst_n low for that cycle instead of checking the output.

gt_rc has no polarity branch (polarity_required=False, no self.polarity use),
so there is a single bare module gt_rc.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_gt_rc.py
"""
import sys
from pathlib import Path

import torch
from napl.operation import gt_rc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "gt_rc.vec"

# test_gt_rc.py codec_config1/2: the two encoders feeding gt_rc, distinct dims.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def main():
    model = gt_rc(config={})
    model.reset()

    rows = []  # (rst_n, in_0, in_1, out)

    # All representative operand pairs (corners + sweep + draws), split into two
    # batches so a reset can be injected between them while state is dirtied.
    pairs = rep_pairs("bipolar", "bipolar")
    half = len(pairs) // 2
    batch_a, batch_b = pairs[:half], pairs[half:]

    # Batch A: drive from the fresh reset above, recording each cycle.
    s0, s1 = pair_streams(CODEC0, CODEC1, batch_a)
    for a, b in zip(s0, s1):
        out = int(model(torch.tensor(a), torch.tensor(b)).item())
        rows.append((1, a, b, out))

    # MID-STREAM reset: state is now dirtied by batch A. reset() returns the
    # model to its post-reset state (dff->1, cnt->0). Emit a reset-pulse row.
    model.reset()
    rows.append((0, 0, 0, 0))

    # Batch B: continue after the reset; outputs must match a model reset from a
    # dirtied state, proving RTL i_rst_n equivalence.
    s0, s1 = pair_streams(CODEC0, CODEC1, batch_b)
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
