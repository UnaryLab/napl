"""
Generate golden test vectors for the sqrt_tracejkff RTL modules straight from
napl's functional Python model (napl.sim.operation.sqrt_tracejkff) -- so the
testbench checks the Verilog against the *actual* simulator, not a hand-derived
truth table.

sqrt_tracejkff is stateful (a JK-FF trace register; bipolar also carries a
width-2 bi2uni accumulator). reset() zeroes both. We drive each polarity model
from reset() with the SAME input stream and record (in, out_unipolar,
out_bipolar) per cycle. The output at cycle t is what forward() returns at that
timestep, which is combinational in i_in given the cycle's incoming state.

Because the op is stateful, we also emit a MID-STREAM reset: after dirtying the
trace/acc registers with a leading segment, we call model.reset() and emit a
sentinel "R" row, then keep driving. The testbench pulses active-low i_rst_n on
the "R" row and must match the model again from the freshly-cleared state, which
proves the RTL reset is equivalent to the Python reset() from a dirty state.

Output: ../vec/sqrt_tracejkff.vec, one line per cycle:

    <in> <out_unipolar> <out_bipolar>   (each 0/1, space-separated)

or the sentinel reset row:

    R

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sqrt_tracejkff.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import sqrt_tracejkff

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "sqrt_tracejkff.vec"

# test_sqrt_tracejkff.py codec_config: bipolar encoder, but the test draws
# unipolar-range [0,1] operands (sqrt input is non-negative).
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def clear_state(uni, bip):
    """Clear exactly the registers the RTL i_rst_n clears (trace=0, acc=0).

    sqrt_tracejkff.reset() only zeroes timestep_cur, not the JK-FF trace or the
    bi2uni accumulator, so to mirror the RTL active-low reset we reset the
    sub-modules directly.
    """
    uni.reset()
    bip.reset()
    uni.jkff.reset()
    bip.jkff.reset()
    bip.bi2uni.reset()


def emit(f, uni, bip, bit):
    in_u = torch.tensor(bit, dtype=uni.stype)
    in_b = torch.tensor(bit, dtype=bip.stype)
    out_u = int(uni(in_u).item())
    out_b = int(bip(in_b).item())
    f.write(f"{bit} {out_u} {out_b}\n")


def main():
    uni = sqrt_tracejkff(config={"polarity": "unipolar"})
    bip = sqrt_tracejkff(config={"polarity": "bipolar"})

    # the test's encoder streams for representative operands, concatenated.
    values = rep_values(CODEC["polarity"], value_range=(0.0, 1.0))
    # split into a leading "dirtying" segment and a post-reset segment so the
    # mid-stream reset is exercised from a non-zero trace/acc state.
    split = max(1, len(values) // 2)
    head_vals, tail_vals = values[:split], values[split:]

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        # Phase 1: drive from a clean reset to dirty the stateful registers.
        clear_state(uni, bip)
        for v in head_vals:
            for bit in encode_value(CODEC, v):
                emit(f, uni, bip, bit)
                rows += 1
        # Mid-stream reset: clear the model state and emit the sentinel row the
        # testbench turns into an i_rst_n pulse. Proves RTL reset == Python
        # reset from a dirtied state.
        clear_state(uni, bip)
        f.write("R\n")
        rows += 1
        # Phase 2: continue from the freshly-cleared state.
        for v in tail_vals:
            for bit in encode_value(CODEC, v):
                emit(f, uni, bip, bit)
                rows += 1
    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()
