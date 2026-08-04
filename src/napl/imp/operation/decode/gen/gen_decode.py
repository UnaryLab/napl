"""
Generate golden test vectors for the decode RTL straight from napl's functional
Python model (napl.sim.operation.decode) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

decode is a stateful accumulator: reset() clears the count, and each timestep adds
the incoming spike. The model asserts timestep_cur <= timestep, so each encoded
operand stream is preceded by a reset(); that reset also proves the RTL's
active-low i_rst_n reload matches the model from a dirtied counter state.

Output: ../vec/decode.vec, one line per timestep:

    <i_spike> <o_spike_count> <rst>   (spike and rst are 0/1, count is decimal;
                                       rst=1 means "reset BEFORE this cycle")

The sizing param WIDTH is the single source of truth: it is derived by the model
from config['timestep'], used to build the model, AND emitted into
../vec/decode_params.vh as `GEN_WIDTH so the testbench overrides the RTL
parameter with the same value. RTL and sim therefore inherit WIDTH from one
place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_decode.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import decode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "decode.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "decode_params.vh"

# Sizing and encoder settings mirror test_decode.py's rank-two case.
DECODE = {"polarity": "bipolar", "timestep": 16}
CODEC = {"polarity": "bipolar", "timestep": 16, "generator": "sobol", "dim": 1}


def main():
    model = decode(config=DECODE)
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_WIDTH {model.width}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    rows = 0
    with VEC.open("w") as f:
        for vi, v in enumerate(rep_values(CODEC["polarity"])):
            for si, bit in enumerate(encode_value(CODEC, v)):
                rst = 0
                if vi > 0 and si == 0:
                    model.reset()
                    rst = 1
                model(torch.tensor([bit], dtype=model.stype))
                count = int(model.spike_count.item())
                f.write(f"{bit} {count} {rst}\n")
                rows += 1
    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} (GEN_WIDTH={model.width})")


if __name__ == "__main__":
    main()
