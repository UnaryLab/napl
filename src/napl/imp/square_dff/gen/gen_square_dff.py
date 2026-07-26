"""
Generate golden test vectors for the square_dff RTL straight from napl's
functional Python model (napl.sim.operation.square_dff) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

square_dff is stateful: it squares one spike stream by AND/XNOR-ing it with a
depth-DEPTH delayed copy of itself (an embedded dff). The delay line is reset to
all-zeros by reset(), so we drive a multi-cycle stream from a fresh reset and
record the per-cycle (input, output) for both polarity variants. To prove reset
equivalence from a *dirtied* state we also inject a MID-STREAM reset (rst=1):
the model.reset() is replayed there and the RTL's active-low i_rst_n is pulsed,
and both must resume from the cleared delay line identically.

The sizing param DEPTH is the single source of truth here: it is read ONCE from
the op config (mirroring test_square_dff.py's square_dff_config), used to build
the model, AND emitted into ../vec/square_dff_params.vh as `GEN_DEPTH so the
testbench overrides the RTL parameter with the same value. RTL and sim therefore
inherit DEPTH from one place; they cannot drift.

Output: ../vec/square_dff.vec, one line per timestep:

    <rst> <i_input> <out_unipolar> <out_bipolar>   (each 0/1, space-separated)

rst=1 marks cycles where reset() is applied (to the model) / i_rst_n pulsed low
(in the RTL) BEFORE driving i_input.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_square_dff.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import square_dff

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "square_dff.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "square_dff_params.vh"

# test_square_dff.py square_dff_config: the sizing param the op is built with.
# (depth is consumed by square_dff's embedded dff to size the delay line.)
SQUARE_DFF = {"polarity": "bipolar", "depth": 1}
# test_square_dff.py codec_config: the encoder feeding square_dff.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}

# Representative operands, encoded with the test's exact codec. We split the
# operands into two halves; a mid-stream reset is injected at the boundary so the
# delay line is dirty when reset() fires, proving reset equivalence.
_VALUES = rep_values(CODEC["polarity"])
_MID = len(_VALUES) // 2

# Build the (reset_flag, spike) drive list.
DRIVE = []
for _i, _v in enumerate(_VALUES):
    _seg = encode_value(CODEC, _v)
    for _j, _s in enumerate(_seg):
        # First cycle of the mid-point segment: reset before driving.
        _rst = 1 if (_i == _MID and _j == 0) else 0
        DRIVE.append((_rst, _s))


def run(polarity):
    model = square_dff(config={"polarity": polarity, "depth": SQUARE_DFF["depth"]})
    model.reset()
    outs = []
    for rst, s in DRIVE:
        if rst:
            model.reset()
        spike = torch.tensor(s, dtype=model.stype)
        outs.append(int(model(spike).item()))
    return outs


def main():
    out_uni = run("unipolar")
    out_bi = run("bipolar")

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    model = square_dff(config=dict(SQUARE_DFF))
    PARAMS.write_text(
        f"`define GEN_DEPTH {SQUARE_DFF['depth']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    with VEC.open("w") as f:
        for (rst, s), u, b in zip(DRIVE, out_uni, out_bi):
            f.write(f"{rst} {s} {u} {b}\n")
    print(
        f"wrote {VEC} ({len(DRIVE)} vectors) and {PARAMS} "
        f"(GEN_DEPTH={SQUARE_DFF['depth']}, GEN_PP_DELAY={model.hw.pp_delay})"
    )


if __name__ == "__main__":
    main()
