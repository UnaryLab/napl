"""
Generate golden vectors for the sub_scale RTL module from napl's Python model
(napl.sim.operation.sub_scale), so the testbench checks the Verilog against the
actual simulator rather than a hand-derived truth table.

Output: ../vec/sub_scale.vec, one line per timestep:

    <rst> <a> <b> <output>      (each 0/1, space-separated)

sub_scale is bipolar only: it negates the subtrahend b and reduces {a, -b} with
a scaled adder, so (a - b) / scale == (a + (-b)) / scale. It carries the
add_scale accumulator, so each row is one cycle of an independent reset segment;
<rst> marks the first cycle of each segment (i_rst_n pulsed low, acc <= 0).

Config mirrors tests/operation/test_sub_scale.py (scale 2, intwidth 20) with the
fidelity a/b sweep narrowed to |a - b| <= scale so (a - b) / scale stays inside
the bipolar range. The sim's fractional grid has no hardware form, so the RTL
runs the add_scale integer grid (fracwidth 0), the value mapping.yaml pins; the
sweep here uses fracwidth 0 to match.

Run inside the `napl` conda env:
    python gen/gen_sub_scale.py
"""
from pathlib import Path

import torch

from napl.sim.operation import encode, sub_scale
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "sub_scale.vec"
PARAMS = ROOT / "vec" / "sub_scale_params.vh"

# Mirrors test_sub_scale.py's scale and intwidth; fracwidth is 0 because the RTL
# accumulator counts whole spikes (the add_scale integer grid).
TIMESTEP = 256
CONFIG = {"polarity": "bipolar", "scale": 2, "intwidth": 20, "fracwidth": 0}
# Fidelity sweep from test_sub_scale.make_values, narrowed to keep the difference
# strictly inside the bipolar range at scale 2.
SWEEP = 128


def encode_segments(a_values, b_values):
    """Encode each (a, b) pair as one independent scalar RTL circuit's streams.

    a and b are independent streams, so each pair gets its own encoders on
    distinct Sobol dimensions (decorrelated); the model is fed the same spikes
    the vector records, so bit-exactness never depends on that correlation.
    """
    segments = []
    for a_value, b_value in zip(a_values, b_values):
        enc_a = encode({"polarity": "bipolar", "timestep": TIMESTEP,
                        "generator": "sobol", "dim": 1})
        enc_b = encode({"polarity": "bipolar", "timestep": TIMESTEP,
                        "generator": "sobol", "dim": 2})
        enc_a.reset()
        enc_b.reset()
        a_scalar = torch.tensor(float(a_value))
        b_scalar = torch.tensor(float(b_value))
        segments.append([
            (enc_a(a_scalar).clone(), enc_b(b_scalar).clone())
            for _ in range(TIMESTEP)
        ])
    return segments


def run(segments):
    """Generate bit-exact Python outputs, resetting before each independent row."""
    model = sub_scale(dict(CONFIG))
    outputs = []
    for segment in segments:
        model.reset()
        outputs.append([
            int(model(a_bit, b_bit).item()) for a_bit, b_bit in segment
        ])
    return outputs, model.hw.pp_delay


def main():
    a_values = torch.linspace(-0.9, 0.9, SWEEP)
    b_values = torch.linspace(0.9, -0.9, SWEEP)
    segments = encode_segments(a_values, b_values)
    outputs, pp_delay = run(segments)

    # Resolve the elaborated mapping entry and require it to reproduce the params
    # these vectors were built with, so make test gates the mapping too.
    binding = translate_node({"class": "sub_scale", "config": {
        "scale": CONFIG["scale"], "intwidth": CONFIG["intwidth"],
        "fracwidth": CONFIG["fracwidth"]}})
    expected = {"SCALE": CONFIG["scale"], "WIDTH": CONFIG["intwidth"]}
    assert binding.parameters == expected, \
        f"mapping sub_scale resolves {binding.parameters}, not {expected}"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALE {CONFIG['scale']}\n"
        f"`define GEN_WIDTH {CONFIG['intwidth']}\n"
        f"`define GEN_PP_DELAY {pp_delay}\n"
    )

    with VEC.open("w") as output:
        for segment_index, segment in enumerate(segments):
            for cycle, (a_bit, b_bit) in enumerate(segment):
                reset = int(cycle == 0)
                output.write(
                    f"{reset} {int(a_bit.item())} {int(b_bit.item())} "
                    f"{outputs[segment_index][cycle]}\n"
                )

    vector_count = sum(len(segment) for segment in segments)
    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments)} reset segments) "
        f"and {PARAMS} (SCALE={CONFIG['scale']} WIDTH={CONFIG['intwidth']} "
        f"PP_DELAY={pp_delay})"
    )


if __name__ == "__main__":
    main()
