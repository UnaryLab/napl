"""
Generate golden test vectors for the sqrt_traceiscb RTL modules straight from
napl's functional Python model (napl.operation.sqrt_traceiscb) -- so the
testbenches check the Verilog against the *actual* simulator, not a hand-derived
truth table.

sqrt_traceiscb is a stateful bit-serial square-root circuit (stochastic bit
inserting via the in-stream correlation-based division cordiv kernel). It has two
polarity variants (unipolar / bipolar) with independent state, so we drive BOTH
models from reset() with the same deterministic 0/1 stream and record per cycle:

    ../vec/sqrt_traceiscb.vec, one line per cycle:

        <rst> <in> <out_unipolar> <out_bipolar>   (each 0/1, space-separated)

The output at cycle t is what forward() returns at that timestep. <rst> is a
reset marker: when 1, the cycle applies the model's reset() BEFORE driving the
input, and the recorded outputs are those of the freshly-reset model. We inject a
mid-stream reset partway through (after the streams have dirtied every register)
so the co-sim proves the RTL's active-low i_rst_n reproduces reset() from an
arbitrary dirtied state, not just at t=0.

sqrt_traceiscb has no config-derived sizing param: its only config key is
'polarity'. The cordiv depth (2) and bi2uni width (2) are intrinsic algorithm
constants fixed inside the op's __init__ ("the config is fixed to optimal
directly"), not derived from the op's own config -- so no <op>_params.vh is
emitted and the RTL carries no parameter.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sqrt_traceiscb.py
"""
import sys
from pathlib import Path

import torch
from napl.operation import sqrt_traceiscb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "sqrt_traceiscb.vec"

# test_sqrt_traceiscb.py codec_config: bipolar encoder, but the test draws
# unipolar-range [0,1] operands (sqrt input is non-negative).
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def main():
    uni = sqrt_traceiscb(config={"polarity": "unipolar"})
    bi = sqrt_traceiscb(config={"polarity": "bipolar"})
    uni.reset()
    bi.reset()

    # the test's encoder streams for representative operands, concatenated.
    stream = []
    for v in rep_values(CODEC["polarity"], value_range=(0.0, 1.0)):
        stream += encode_value(CODEC, v)

    # Inject a mid-stream reset once the state is thoroughly dirtied, then keep
    # streaming so the post-reset trajectory is also checked.
    reset_at = len(stream) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for i, bit in enumerate(stream):
            do_reset = 1 if i == reset_at else 0
            if do_reset:
                uni.reset()
                bi.reset()
            inp_u = torch.tensor(bit, dtype=uni.stype)
            inp_b = torch.tensor(bit, dtype=bi.stype)
            out_uni = int(uni(inp_u).item())
            out_bi = int(bi(inp_b).item())
            f.write(f"{do_reset} {bit} {out_uni} {out_bi}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors, mid-stream reset at cycle {reset_at})")


if __name__ == "__main__":
    main()
