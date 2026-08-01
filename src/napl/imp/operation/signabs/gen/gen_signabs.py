"""
Generate golden test vectors for the signabs RTL module straight from napl's
functional Python model (napl.sim.operation.signabs) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

signabs is stateful: a saturating up/down accumulator (reset to ACC_MED =
2**(WIDTH-1), NOT zero) whose UPDATED value (after folding in the current input)
sets the sign/abs outputs in the same timestep. We drive the model from reset()
with a known input stream and record (input, sign, abs) every cycle.

Output: ../vec/signabs.vec, one line per cycle:

    <in> <sign> <abs>     (each 0/1, space-separated)

The sizing param WIDTH is the single source of truth here: it is read from the
op config (mirroring test_signabs.py's signabs_config), used to build the
model, AND emitted into ../vec/signabs_params.vh as `GEN_WIDTH so the testbench
overrides the RTL parameter with the same value. RTL and sim therefore inherit
WIDTH from one place; they cannot drift.

signabs is stateful, so the stream includes a MID-STREAM reset() (and a reset
marker column) to prove the RTL reproduces the post-reset() accumulator state
(acc = ACC_MED) from a dirtied state, not just from t=0.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_signabs.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import signabs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "signabs.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "signabs_params.vh"

# Sizing and encoder settings mirror test_signabs.py.
SIGNABS = {"width": 3}
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def main():
    model = signabs(config=SIGNABS)
    model.reset()

    # reset_flag requests reset before the corresponding timestep.
    values = rep_values(CODEC["polarity"])
    mid = len(values) // 2
    items = []
    for idx, v in enumerate(values):
        bits = encode_value(CODEC, v)
        for j, bit in enumerate(bits):
            reset_flag = 1 if (idx == mid and j == 0) else 0
            items.append((bit, reset_flag))

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_WIDTH {SIGNABS['width']}\n")

    rows = 0
    with VEC.open("w") as f:
        for bit, reset_flag in items:
            if reset_flag:
                model.reset()
            inp = torch.tensor(bit, dtype=model.stype)
            sign, abs_ = model(inp)
            f.write(f"{reset_flag} {bit} {int(sign.item())} {int(abs_.item())}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors, WIDTH={SIGNABS['width']}) and {PARAMS}")


if __name__ == "__main__":
    main()
