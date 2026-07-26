"""
Generate golden test vectors for the relu_cnt RTL straight from napl's functional
Python model (napl.sim.operation.relu_cnt) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

relu_cnt is stateful (a saturating up/down counter), so the stream order matters:
the model is reset() once at t=0 and then driven cycle by cycle, recording the
output each cycle. We also inject a MID-STREAM reset() to prove the RTL's
active-low i_rst_n reload (acc = HALF) matches the model's reset() from a dirtied
counter state. Reset events are recorded as a third column so the testbench can
pulse i_rst_n at the matching cycle.

Output: ../vec/relu_cnt.vec, one line per timestep:

    <i_input> <o_out> <rst>   (each 0/1, space-separated; rst=1 means "reset BEFORE
                            this cycle's output is produced")

The sizing param WIDTH is the single source of truth: it is read from the op
config (mirroring test_relu_cnt.py's relu_cnt_config), used to build the model,
AND emitted into ../vec/relu_cnt_params.vh as `GEN_WIDTH so the testbench
overrides the RTL parameter with the same value. RTL and sim therefore inherit
WIDTH from one place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_relu_cnt.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import relu_cnt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "relu_cnt.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "relu_cnt_params.vh"

# test_relu_cnt.py relu_cnt_config: the sizing param the op is built with.
RELU_CNT = {"width": 3}
# test_relu_cnt.py codec_config: the encoder feeding relu_cnt.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def main():
    model = relu_cnt(config=RELU_CNT)
    model.reset()

    # the test's encoder streams for representative operands, concatenated.
    vals = rep_values(CODEC["polarity"])
    # Pick a mid-stream reset point: after the first operand's stream, reset the
    # model (acc dirtied by the first operand) and continue. The RTL pulses
    # i_rst_n at that cycle so both reload acc = HALF and must agree afterward.
    reset_after_value_idx = 1  # reset before encoding the 2nd representative value

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    PARAMS.write_text(f"`define GEN_WIDTH {RELU_CNT['width']}\n")

    rows = 0
    with VEC.open("w") as f:
        for vi, v in enumerate(vals):
            spikes = encode_value(CODEC, v)
            for si, bit in enumerate(spikes):
                rst = 0
                # mid-stream reset: at the first cycle of the chosen operand,
                # reset the model and tell the tb to pulse i_rst_n first.
                if vi == reset_after_value_idx and si == 0:
                    model.reset()
                    rst = 1
                x = torch.tensor(bit, dtype=model.stype)
                out = int(model(x).item())
                f.write(f"{bit} {out} {rst}\n")
                rows += 1
    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} (GEN_WIDTH={RELU_CNT['width']})")


if __name__ == "__main__":
    main()
