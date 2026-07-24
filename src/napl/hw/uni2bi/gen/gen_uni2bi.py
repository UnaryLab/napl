"""
Generate golden test vectors for the uni2bi RTL module straight from napl's
functional Python model (napl.sim.operation.uni2bi) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/uni2bi.vec, one line per cycle:

    <i_in> <o_out>        (each 0/1, space-separated)

uni2bi is stateful (a signed accumulator), so a single input stream is driven
through the model cycle by cycle from reset() and the per-cycle (input, output)
pair is recorded. The input spikes are the SAME ones tests/operation/test_uni2bi.py
sends the op: representative operand values encoded by the test's encoder config
(codec_config1: unipolar / timestep=256 / sobol / dim=1), concatenated into one
continuous stream. A MID-STREAM reset() is injected partway through to prove the
RTL's active-low reset reproduces the model's reset() from a dirtied accumulator
state, not just from t=0.

The sizing param WIDTH is the single source of truth here: it is read from the
op config (mirroring test_uni2bi.py's uni2bi_config), used to build the model,
AND emitted into ../vec/uni2bi_params.vh as `GEN_WIDTH so the testbench overrides
the RTL parameter with the same value. RTL and sim therefore inherit WIDTH from
one place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_uni2bi.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import uni2bi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "uni2bi.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "uni2bi_params.vh"

# test_uni2bi.py uni2bi_config: the sizing param the op is built with.
UNI2BI = {"width": 3}
# test_uni2bi.py codec_config1: the encoder feeding uni2bi.
CODEC = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}

# Special sentinel emitted to drive a mid-stream reset (asserts active-low reset
# from a dirtied accumulator). It carries no (in,out) vector; the tb sees the
# "R" marker, pulses i_rst_n low, and the model calls reset().
RESET_MARK = "R"


def build_segments():
    """List of per-operand spike segments (each a list of 0/1), test order."""
    return [encode_value(CODEC, v) for v in rep_values(CODEC["polarity"])]


def main():
    model = uni2bi({"width": UNI2BI["width"]})
    model.reset()

    segments = build_segments()
    # inject a reset halfway through the segment list to dirty-then-reset the acc.
    reset_after = len(segments) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_WIDTH {UNI2BI['width']}\n")

    rows = 0
    with VEC.open("w") as f:
        for seg_idx, seg in enumerate(segments):
            if seg_idx == reset_after:
                model.reset()
                f.write(f"{RESET_MARK}\n")
            for bit in seg:
                i_in = torch.tensor(bit, dtype=model.stype)
                o_out = int(model(i_in).item())
                f.write(f"{bit} {o_out}\n")
                rows += 1
    print(f"wrote {VEC} ({rows} vectors, mid-stream reset) "
          f"and {PARAMS} (GEN_WIDTH={UNI2BI['width']})")


if __name__ == "__main__":
    main()
