from pathlib import Path

import torch

from napl.sim.module import encoder
from napl.sim.operation import add_any


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_any.vec"
PARAMS = ROOT / "vec" / "add_any_params.vh"

# These are the primary regimes in tests/operation/test_add_any.py.
TIMESTEP = 256
ADD_ANY = {"scale": 8, "width": 20}
ENTRY = ADD_ANY["scale"]


def test_values(polarity):
    """Return the test rows in the requested, probability-equivalent polarity."""
    known = torch.full((8, ENTRY), 0.25)
    fidelity = torch.linspace(-0.75, 0.75, 512).reshape(64, ENTRY)
    values = torch.cat((known, fidelity), dim=0)
    if polarity == "unipolar":
        values = (values + 1) / 2
    return values


def encode_segments(polarity, values):
    """Encode each test row as one independent scalar RTL circuit's input stream."""
    segments = []
    for values_row in values:
        enc = encoder({
            "polarity": polarity,
            "timestep": TIMESTEP,
            "generator": "sobol",
            "dim": 1,
        })
        enc.reset()
        segments.append([enc(values_row).clone() for _ in range(TIMESTEP)])
    return segments


def run(polarity, segments):
    """Generate bit-exact Python outputs and reset before every independent row."""
    model = add_any({"polarity": polarity, **ADD_ANY})
    outputs = []
    for segment in segments:
        model.reset()
        outputs.append([int(model(spikes, dim=-1).item()) for spikes in segment])
    return outputs, model.hw.pp_delay


def bus(spikes):
    """Format lane 0 as the Verilog vector's least-significant bit."""
    return "".join(str(int(bit)) for bit in reversed(spikes.tolist()))


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar"))
    segments_bi = encode_segments("bipolar", test_values("bipolar"))
    assert len(segments_uni) == len(segments_bi)
    out_uni, pp_delay_uni = run("unipolar", segments_uni)
    out_bi, pp_delay_bi = run("bipolar", segments_bi)
    assert pp_delay_uni == pp_delay_bi

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALE {ADD_ANY['scale']}\n"
        f"`define GEN_WIDTH {ADD_ANY['width']}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_PP_DELAY {pp_delay_uni}\n"
    )

    with VEC.open("w") as output:
        for segment_index, (segment_uni, segment_bi) in enumerate(
            zip(segments_uni, segments_bi)
        ):
            for cycle, (spikes_uni, spikes_bi) in enumerate(
                zip(segment_uni, segment_bi)
            ):
                reset = int(cycle == 0)
                output.write(
                    f"{reset} {bus(spikes_uni)} {out_uni[segment_index][cycle]} "
                    f"{bus(spikes_bi)} {out_bi[segment_index][cycle]}\n"
                )

    vector_count = len(segments_uni) * TIMESTEP
    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments_uni)} reset segments) "
        f"and {PARAMS} (SCALE={ADD_ANY['scale']} WIDTH={ADD_ANY['width']} "
        f"ENTRY={ENTRY} PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
