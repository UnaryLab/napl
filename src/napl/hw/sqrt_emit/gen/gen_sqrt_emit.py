"""
Generate golden test vectors for the sqrt_emit RTL straight from napl's
functional Python model (napl.sim.operation.sqrt_emit) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

sqrt_emit is opportunistic-bit-inserting square root. It is stateful: a unipolar
non-scaled accumulator (add_any, scale=1, width=3), a depth-2 shift register that
scrambles the inverted output, the emitted feedback bit, and -- for bipolar only
-- a bi2uni accumulator (width=2). Both polarity variants share the same input
stream and the same accumulator/shiftreg path; only the feedback (emit) differs
(unipolar uses output directly, bipolar uses bi2uni(output)).

These sizes (nsadd width=3, shiftreg depth=2, bi2uni width=2) are intrinsic
algorithm constants hardcoded inside sqrt_emit.__init__; the op's only config key
is `polarity` (no config-derived numeric size key), so the RTL carries no sizing
parameters and is validated as-is.

We drive both models from reset() with one shared input stream and record, per
cycle, the input, a mid-stream RESET flag, and each variant's output. The output
at cycle t is what forward() returns at that timestep (combinational in i_in
given the cycle-t registers). To prove reset equivalence from a *dirtied* state
(both models are stateful) we inject a mid-stream reset(): at that cycle the RTL
re-asserts i_rst_n and both models are reset() before resuming the same stream.

Output: ../vec/sqrt_emit.vec, one line per cycle:

    <in> <rst> <out_unipolar> <out_bipolar>   (each 0/1, space-separated)

where rst=1 means "apply reset() at the START of this cycle, then process this
cycle's input from the post-reset state".

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sqrt_emit.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import sqrt_emit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "sqrt_emit.vec"

# test_sqrt_emit.py codec_config: the encoder feeding sqrt_emit.
CODEC = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 4}


def main():
    uni = sqrt_emit({"polarity": "unipolar"})
    bip = sqrt_emit({"polarity": "bipolar"})
    uni.reset()
    bip.reset()

    # the test's encoder streams for representative operands, concatenated.
    # mark the FIRST cycle of each operand segment except the very first so we
    # can inject a mid-stream reset partway through the stream (dirtied state).
    stream = []          # list of (bit, rst_flag)
    vals = rep_values(CODEC["polarity"])
    # pick a reset point well into the stream (start of the 3rd operand segment)
    reset_seg = 2 if len(vals) > 2 else len(vals) - 1
    for si, v in enumerate(vals):
        seg = encode_value(CODEC, v)
        for bi, bit in enumerate(seg):
            rst = 1 if (si == reset_seg and bi == 0) else 0
            stream.append((bit, rst))

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for bit, rst in stream:
            if rst:
                uni.reset()
                bip.reset()
            inp_u = torch.tensor(int(bit), dtype=uni.stype)
            inp_b = torch.tensor(int(bit), dtype=bip.stype)
            out_u = int(uni(inp_u).item())
            out_b = int(bip(inp_b).item())
            f.write(f"{int(bit)} {rst} {out_u} {out_b}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors, mid-stream reset at segment {reset_seg})")


if __name__ == "__main__":
    main()
