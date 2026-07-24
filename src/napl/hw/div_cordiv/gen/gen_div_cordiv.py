"""
Generate golden test vectors for the div_cordiv RTL module straight from napl's
functional Python model (napl.sim.operation.div_cordiv) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

div_cordiv is unipolar-only (polarity_required=False) correlated division. It is
stateful: a depth-DEPTH circular buffer of past quotients plus a cyclic index
idx. We mirror test_div_cordiv.py: both dividend and divisor are encoded on the
SAME sobol dim (dim=1) -> correlated streams, and the test sorts so the dividend
<= divisor (proper-fraction quotient) with a non-zero divisor.

The sizing param DEPTH is the single source of truth: it is read from the op
config (mirroring test_div_cordiv.py's div_cordiv_config), used to build the
model, AND emitted into ../vec/div_cordiv_params.vh as `GEN_DEPTH so the testbench
overrides the RTL parameter with the same value. RTL and sim therefore inherit
DEPTH from one place; they cannot drift.

Because div_cordiv is stateful, we also assert a MID-STREAM reset: after dirtying
the buffer with one stream segment we call model.reset() and emit a reset marker
("R"), then keep driving. The testbench re-pulses i_rst_n on that marker and
requires the post-reset outputs to keep matching, proving reset returns BOTH
sides to the same state from an arbitrary (dirtied) operating state.

Output: ../vec/div_cordiv.vec, one line per cycle:

    <dividend> <divisor> <quotient>      (each 0/1, space-separated)

or a lone "R" line marking a mid-stream reset (no check on that line).

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_div_cordiv.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import div_cordiv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "div_cordiv.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "div_cordiv_params.vh"

# test_div_cordiv.py div_cordiv_config: the sizing param the op is built with.
DIV_CORDIV = {"depth": 2, "generator": "Sobol"}
# test_div_cordiv.py codec_config1/2: both unipolar on the SAME sobol dim (dim=1)
# -> correlated dividend/divisor streams, exactly as the test requires. The test
# also sorts so dividend <= divisor (proper-fraction quotient) and divisor != 0.
CODEC0 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def proper_pairs():
    """Representative (dividend, divisor) pairs with 0 < dividend <= divisor,
    mirroring the test's sort + non-zero-divisor masking."""
    pairs = []
    for x, y in rep_pairs("unipolar", "unipolar"):
        lo, hi = (x, y) if x <= y else (y, x)
        hi = hi if hi > 0 else 1.0
        pairs.append((lo, hi))
    return pairs


def run_segment(model, f, pairs):
    """Drive one independent stream segment through the model and write its
    (dividend, divisor, quotient) vectors. Returns the number of rows written."""
    dividend_stream, divisor_stream = pair_streams(CODEC0, CODEC1, pairs)
    rows = 0
    for dv, ds in zip(dividend_stream, divisor_stream):
        dividend = torch.tensor(dv, dtype=model.stype)
        divisor = torch.tensor(ds, dtype=model.stype)
        q = int(model(dividend, divisor).item())
        f.write(f"{dv} {ds} {q}\n")
        rows += 1
    return rows


def main():
    model = div_cordiv(config=dict(DIV_CORDIV))

    pairs = proper_pairs()
    # Split into two segments to exercise a mid-stream reset between them.
    half = len(pairs) // 2
    seg_a, seg_b = pairs[:half], pairs[half:]

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    PARAMS.write_text(f"`define GEN_DEPTH {DIV_CORDIV['depth']}\n")

    rows = 0
    with VEC.open("w") as f:
        # Power-on reset, then a first segment that dirties the buffer.
        model.reset()
        rows += run_segment(model, f, seg_a)
        # Mid-stream reset from the dirtied state: marker for the tb to re-pulse
        # i_rst_n, and model.reset() so both sides return to the reset() state.
        f.write("R\n")
        model.reset()
        rows += run_segment(model, f, seg_b)

    print(f"wrote {VEC} ({rows} vectors, DEPTH={DIV_CORDIV['depth']}) and {PARAMS}")


if __name__ == "__main__":
    main()
