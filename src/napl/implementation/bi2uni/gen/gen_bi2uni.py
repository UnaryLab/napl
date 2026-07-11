"""
Generate golden test vectors for the bi2uni RTL module straight from napl's
functional Python model (napl.operation.bi2uni) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/bi2uni.vec, one line per cycle:

    <i_in> <o_out>        (each 0/1, space-separated)

bi2uni is stateful (a signed accumulator), so a single input stream is driven
through the model cycle by cycle from reset() and the per-cycle (input, output)
pair is recorded. The input spikes are the SAME ones tests/operation/test_bi2uni.py
sends the op: representative operand values encoded by the test's encoder config
(codec_config1: bipolar / timestep=256 / sobol / dim=1), concatenated into one
continuous stream. Because the op is stateful, a MID-STREAM reset() is injected
to prove the RTL's active-low reset reproduces the model's reset() from a dirtied
accumulator state, not just from t=0.

The sizing param WIDTH is the single source of truth here: it is read from the
op config (mirroring test_bi2uni.py's bi2uni_config), used to build the model,
AND emitted into ../vec/bi2uni_params.vh as `GEN_WIDTH so the testbench overrides
the RTL parameter with the same value. RTL and sim therefore inherit WIDTH from
one place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_bi2uni.py
"""
import sys
from pathlib import Path

import torch
from napl.operation import bi2uni

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "bi2uni.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "bi2uni_params.vh"

# test_bi2uni.py bi2uni_config: the sizing param the op is built with.
BI2UNI = {"width": 2}
# test_bi2uni.py codec_config1: the encoder feeding bi2uni.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}

# Marker emitted as the i_in field to tell the testbench to pulse i_rst_n low and
# restart from the model's post-reset() state, then resume checking. Chosen
# outside the {0,1} spike alphabet so it can never collide with a real vector.
RST = "R"


def build_segments():
    """Two stream segments (driven through one fresh-reset model each), so a
    mid-stream reset() between them proves reset equivalence from a dirty acc."""
    half = len(rep_values(CODEC["polarity"])) // 2
    vals = rep_values(CODEC["polarity"])
    return [vals[:half] or vals, vals[half:] or vals]


def main():
    model = bi2uni({"width": BI2UNI["width"]})

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    PARAMS.write_text(f"`define GEN_WIDTH {BI2UNI['width']}\n")

    rows = 0
    with VEC.open("w") as f:
        for seg_idx, seg in enumerate(build_segments()):
            if seg_idx > 0:
                # mid-stream reset marker: RTL pulses i_rst_n low here.
                f.write(f"{RST} {RST}\n")
                rows += 1
            # reset() the model at the start of each segment to mirror the RTL.
            model.reset()
            for v in seg:
                for bit in encode_value(CODEC, v):
                    i_in = torch.tensor(bit, dtype=model.stype)
                    o_out = int(model(i_in).item())
                    f.write(f"{bit} {o_out}\n")
                    rows += 1
    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} (GEN_WIDTH={BI2UNI['width']})")


if __name__ == "__main__":
    main()
