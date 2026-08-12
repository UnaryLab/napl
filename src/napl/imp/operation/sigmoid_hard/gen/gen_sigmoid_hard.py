"""
Generate golden test vectors for the sigmoid_hard RTL straight from napl's
functional Python model (napl.sim.operation.sigmoid_hard) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

sigmoid_hard is add_scale(scale=2, intwidth=4) fed the pre-reduced sum (input+1); the
unipolar and bipolar variants are numerically identical (bipolar offset
(entry-scale)/2 = 0), so a single module covers both. It is stateful (an
accumulator), so we drive a multi-cycle input stream and record per-cycle I/O,
starting from model.reset() at t=0, and inject a MID-STREAM reset() to prove the
RTL's i_rst_n reproduces the model's reset from a dirtied accumulator state.

sigmoid_hard's config key_list is ['polarity'] plus the optional `width`, which
sets the add_scale submodule's intwidth and defaults to 4; scale=2 is an intrinsic
hard-sigmoid constant. intwidth sets only the accumulator clamp bounds, and those
are unreachable at every width the model accepts: add_scale rejects width below 3
(scale 2 exceeds its accumulator maximum), and at width 3 the clamp is already
[-4, 3] while the accumulator holds 0 or 1. So width changes no output bit, the
RTL carries no sizing parameter, and the module is validated as-is (no
<op>_params.vh emitted).

Output: vec/sigmoid_hard.vec, one line per cycle:

    <i_rst_n> <i_input> <o_out>   (each 0/1, space-separated)

i_rst_n=0 marks a cycle where the model was reset (accumulator <- 0) *before*
producing that cycle's output; the testbench pulses its active-low reset there.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sigmoid_hard.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import sigmoid_hard

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "sigmoid_hard.vec"

# Encoder settings mirror test_sigmoid_hard.py.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def build_stream():
    """Representative spike stream over the test's value range.

    The model is identical for both polarities (offset 0), so both unipolar- and
    bipolar-range representative operands are encoded with the test's bipolar
    Sobol codec to widen the in-stream value coverage.
    """
    stream = []
    for v in rep_values("bipolar") + rep_values("unipolar"):
        stream += encode_value(CODEC, v)
    return stream


def main():
    model = sigmoid_hard(config={"polarity": "bipolar"})

    stream = build_stream()
    # Reset after the first half, when the accumulator has changed.
    split = len(stream) // 2

    rows = []  # (i_rst_n, i_input, o_out)

    model.reset()
    rst_pending = True  # this cycle is the first after reset()
    for s in stream[:split]:
        out = int(model(torch.tensor(s)).item())
        rows.append((0 if rst_pending else 1, s, out))
        rst_pending = False

    model.reset()
    rst_pending = True
    for s in stream[split:]:
        out = int(model(torch.tensor(s)).item())
        rows.append((0 if rst_pending else 1, s, out))
        rst_pending = False

    VEC.parent.mkdir(parents=True, exist_ok=True)
    with VEC.open("w") as f:
        for rst_n, s, out in rows:
            f.write(f"{rst_n} {s} {out}\n")
    print(f"wrote {VEC} ({len(rows)} vectors, mid-stream reset at cycle {split})")


if __name__ == "__main__":
    main()
