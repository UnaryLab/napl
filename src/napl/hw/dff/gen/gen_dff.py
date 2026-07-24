"""
Generate golden test vectors for the dff RTL module straight from napl's
functional Python model (napl.sim.operation.dff) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

dff is stateful: a depth-DEPTH bit-serial delay line (a FIFO of D flip-flops)
whose reset() state is ALL ZEROS. We drive the model from reset() with the test's
encoded spike stream and record (input, output) every cycle. The output at cycle
t is what forward() returns at that timestep: the oldest cell, read BEFORE the
new input is pushed.

A MID-STREAM reset is injected to prove reset equivalence from a dirtied state:
after driving part of the stream, we call model.reset() (and emit a reset marker
the testbench replays as an i_rst_n pulse), then continue, so the co-sim checks
that RTL and model return to the identical all-zeros state mid-run.

Output: ../vec/dff.vec, one line per cycle:

    <in> <out>        (each 0/1, space-separated)
    R                 (a lone marker line: pulse i_rst_n low here)

The sizing param DEPTH is the single source of truth here: it is read from the
op config (mirroring test_dff.py's dff_config), used to build the model, AND
emitted into ../vec/dff_params.vh as `GEN_DEPTH so the testbench overrides the
RTL parameter with the same value. RTL and sim therefore inherit DEPTH from one
place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_dff.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import dff

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "dff.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "dff_params.vh"

# test_dff.py dff_config: the sizing param the op is built with.
DFF = {"depth": 1}
# test_dff.py codec_config: the encoder feeding dff.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def main():
    model = dff(config=dict(DFF))
    model.reset()

    # the test's encoder streams for representative operands, concatenated.
    stream = []
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)

    # mid-stream reset point: roughly halfway, to dirty the FIFO then prove the
    # model and RTL both return to the all-zeros reset state.
    reset_at = len(stream) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    PARAMS.write_text(f"`define GEN_DEPTH {DFF['depth']}\n")

    rows = 0
    with VEC.open("w") as f:
        for idx, bit in enumerate(stream):
            if idx == reset_at:
                model.reset()
                f.write("R\n")  # testbench replays this as an i_rst_n pulse
            in_spike = torch.tensor(bit, dtype=model.stype)
            out_spike = int(model(in_spike).item())
            f.write(f"{bit} {out_spike}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors, reset@{reset_at}) and {PARAMS} (GEN_DEPTH={DFF['depth']})")


if __name__ == "__main__":
    main()
