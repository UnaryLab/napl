from pathlib import Path

import torch

from napl.sim.operation import add_any, encode


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_any.vec"
PARAMS = ROOT / "vec" / "add_any_params.vh"

# The carry regime of tests/operation/test_add_any.py (scale 2, width 8) over the
# streaming suite's 8-entry reduction. A scale below the entry count gives the
# bipolar offset (entry - scale) / 2 a nonzero value, which is the only way the
# accumulator ever moves down: with scale == entry the offset is 0, every addend
# is non-negative and the negative clamp -2**(width-1) is unreachable. The width
# is 8 so both clamps are within reach of a 256-step segment.
TIMESTEP = 256
ADD_ANY = {"scale": 2, "width": 8}
ENTRY = 8
# Saturation segment: cycles at the polarity's low rail, then at its high rail.
# The low rail holds the partial sum at 0, so the bipolar accumulator falls by
# the offset each cycle onto -2**(width-1); the high rail charges both polarities
# onto 2**(width-1) - 1. How long each accumulator stays on its clamp is what a
# corrupted clamp changes, so the recharge is what makes the excursion visible.
RAIL_DRAIN = 80
RAIL_CHARGE = 110


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
        enc = encode({
            "polarity": polarity,
            "timestep": TIMESTEP,
            "generator": "sobol",
            "dim": 1,
        })
        enc.reset()
        segments.append([enc(values_row).clone() for _ in range(TIMESTEP)])
    return segments


def rail_segment(polarity):
    """Encode the saturation stimulus: the low rail, then the high rail.

    Both rails are constant values, so one encoder run gives a partial sum of 0
    over the drain and ENTRY over the charge.
    """
    low = 0.0 if polarity == "unipolar" else -1.0
    enc = encode({
        "polarity": polarity,
        "timestep": TIMESTEP,
        "generator": "sobol",
        "dim": 1,
    })
    enc.reset()
    values = [low] * RAIL_DRAIN + [1.0] * RAIL_CHARGE
    return [enc(torch.full((ENTRY,), value)).clone() for value in values]


def run(polarity, segments):
    """Generate bit-exact Python outputs and reset before every independent row.

    The accumulator extremes of the last segment, the saturation one, are
    returned with the outputs so the caller can require it to have reached the
    clamps its width sets.
    """
    model = add_any({"polarity": polarity, **ADD_ANY})
    outputs = []
    extremes = (0, 0)
    for segment in segments:
        model.reset()
        row = []
        low = high = 0
        for spikes in segment:
            row.append(int(model(spikes, dim=-1).item()))
            value = int(model.accumulator.item())
            low, high = min(low, value), max(high, value)
        outputs.append(row)
        extremes = (low, high)
    return outputs, model.hw.pp_delay, extremes, model


def bus(spikes):
    """Format lane 0 as the Verilog vector's least-significant bit."""
    return "".join(str(int(bit)) for bit in reversed(spikes.tolist()))


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar"))
    segments_bi = encode_segments("bipolar", test_values("bipolar"))
    segments_uni.append(rail_segment("unipolar"))
    segments_bi.append(rail_segment("bipolar"))
    assert len(segments_uni) == len(segments_bi)
    out_uni, pp_delay_uni, rail_uni, model_uni = run("unipolar", segments_uni)
    out_bi, pp_delay_bi, rail_bi, model_bi = run("bipolar", segments_bi)
    assert pp_delay_uni == pp_delay_bi
    # One carry is subtracted after the clamp, so a clamped accumulator reads back
    # at acc_max - scale; the negative clamp never fires, so it reads back exactly.
    assert rail_uni[1] == model_uni.acc_max - ADD_ANY["scale"], \
        f"the unipolar rail segment peaked at {rail_uni[1]}, short of {model_uni.acc_max}"
    assert rail_bi[1] == model_bi.acc_max - ADD_ANY["scale"], \
        f"the bipolar rail segment peaked at {rail_bi[1]}, short of {model_bi.acc_max}"
    assert rail_bi[0] == model_bi.acc_min, \
        f"the bipolar rail segment bottomed at {rail_bi[0]}, short of {model_bi.acc_min}"
    # Invariant check, not a stimulus check: the negative clamp is unreachable in
    # unipolar, since the offset is 0, so every addend is non-negative and a carry
    # only ever subtracts down to 0. The unipolar rail segment therefore has no low
    # rail to reach, and this assertion cannot catch a dead stimulus; the three
    # assertions above are the stimulus checks. It stays as the runnable form of
    # the induction the add_any_unipolar header states.
    assert rail_uni[0] == 0, f"the unipolar accumulator went negative, to {rail_uni[0]}"

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

    vector_count = sum(len(segment) for segment in segments_uni)
    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments_uni)} reset segments) "
        f"and {PARAMS} (SCALE={ADD_ANY['scale']} WIDTH={ADD_ANY['width']} "
        f"ENTRY={ENTRY} PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
