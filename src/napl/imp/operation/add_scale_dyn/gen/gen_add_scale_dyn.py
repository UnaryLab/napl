import math
from pathlib import Path

import torch

from napl.sim.operation import add_scale_dyn, encode


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_scale_dyn.vec"
PARAMS = ROOT / "vec" / "add_scale_dyn_params.vh"

# Carry regime mirroring test_add_scale_dyn.py's 8-entry reduction on the integer
# grid (fracwidth 0), with scale_max below the entry count so the bipolar offset
# drives the accumulator down and intwidth 6 so both clamps are reachable and
# main() can prove them observable.
TIMESTEP = 256
ADD_SCALE_DYN = {"scale_max": 3, "intwidth": 6, "fracwidth": 0}
ENTRY = 8
# The rail assertions in main() read the accumulator in raw units and compare against
# acc_max and acc_min in value units, which agree only on the integer grid.
assert ADD_SCALE_DYN["fracwidth"] == 0, "rail assertions need the integer grid, fracwidth 0"
SCALE_MAX = ADD_SCALE_DYN["scale_max"]
# Bus width holding a scale in [1, scale_max].
SCALE_W = math.ceil(math.log2(SCALE_MAX + 1))
# Saturation segment (high rail, low rail, high rail): the high rail charges both
# polarities onto the high clamp and the low rail drains the surplus and carries
# the bipolar accumulator to its low clamp, so both phases are needed to make a
# corrupted clamp observable.
RAIL_DRAIN = 100
RAIL_CHARGE = 60
# The rail segment holds the smallest scale, which drains and charges fastest: the
# bipolar offset (entry - scale) / 2 and the unipolar surplus entry - scale are both
# largest there.
RAIL_SCALE = 1


def test_values(polarity):
    """Return the test rows in the requested, probability-equivalent polarity."""
    known = torch.full((8, ENTRY), 0.25)
    fidelity = torch.linspace(-0.75, 0.75, 512).reshape(64, ENTRY)
    values = torch.cat((known, fidelity), dim=0)
    if polarity == "unipolar":
        values = (values + 1) / 2
    return values


def scale_schedule(segment_index, length):
    """Return one segment's per-timestep scales, cycling the whole legal range."""
    return [1 + ((cycle + segment_index) % SCALE_MAX) for cycle in range(length)]


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
    """Encode the saturation stimulus: charge, drain, charge again.

    Both rails are constant values, so one encoder run gives a partial sum of ENTRY
    over each charge and 0 over the drain.
    """
    low = 0.0 if polarity == "unipolar" else -1.0
    enc = encode({
        "polarity": polarity,
        "timestep": TIMESTEP,
        "generator": "sobol",
        "dim": 1,
    })
    enc.reset()
    values = [1.0] * RAIL_CHARGE + [low] * RAIL_DRAIN + [1.0] * RAIL_CHARGE
    return [enc(torch.full((ENTRY,), value)).clone() for value in values]


def schedules(segments):
    """Return the per-timestep scales of every segment, rails held at RAIL_SCALE."""
    plan = [
        scale_schedule(index, len(segment))
        for index, segment in enumerate(segments[:-1])
    ]
    plan.append([RAIL_SCALE] * len(segments[-1]))
    return plan


def run(polarity, segments, plan, intwidth=None):
    """Generate bit-exact Python outputs and reset before every independent row.

    The extreme pre-clamp sums reached over the whole run are returned with the
    outputs, so the caller can require the clamps to have been reached.
    """
    config = dict(ADD_SCALE_DYN)
    if intwidth is not None:
        config["intwidth"] = intwidth
    model = add_scale_dyn({"polarity": polarity, **config})
    outputs = []
    low = high = 0.0
    for segment, schedule in zip(segments, plan):
        model.reset()
        row = []
        for spikes, scale in zip(segment, schedule):
            offset = (ENTRY - scale) / 2 if polarity == "bipolar" else 0.0
            presum = float(model.accumulator.item()) + float(spikes.sum().item()) - offset
            low, high = min(low, presum), max(high, presum)
            row.append(int(model(spikes, scale, dim=-1).item()))
        outputs.append(row)
    return outputs, model.hw.pp_delay, (low, high), model


def bus(spikes):
    """Format lane 0 as the Verilog vector's least-significant bit."""
    return "".join(str(int(bit)) for bit in reversed(spikes.tolist()))


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar"))
    segments_bi = encode_segments("bipolar", test_values("bipolar"))
    segments_uni.append(rail_segment("unipolar"))
    segments_bi.append(rail_segment("bipolar"))
    assert len(segments_uni) == len(segments_bi)
    plan = schedules(segments_uni)
    out_uni, pp_delay_uni, rail_uni, model_uni = run("unipolar", segments_uni, plan)
    out_bi, pp_delay_bi, rail_bi, model_bi = run("bipolar", segments_bi, plan)
    assert pp_delay_uni == pp_delay_bi

    # Stimulus checks: WIDTH is observable only through the clamps, so the run must
    # reach each one that is reachable and produce outputs a wider accumulator would
    # not.
    assert rail_uni[1] > model_uni.acc_max, \
        f"the unipolar pre-clamp sum peaked at {rail_uni[1]}, short of {model_uni.acc_max}"
    assert rail_bi[1] > model_bi.acc_max, \
        f"the bipolar pre-clamp sum peaked at {rail_bi[1]}, short of {model_bi.acc_max}"
    assert rail_bi[0] < model_bi.acc_min, \
        f"the bipolar pre-clamp sum bottomed at {rail_bi[0]}, short of {model_bi.acc_min}"
    # Invariant check, not a stimulus check: the unipolar negative clamp is
    # unreachable (offset 0, so every addend is non-negative and a carry only
    # subtracts to 0), so this is the runnable form of the induction the
    # add_scale_dyn_unipolar header states, not a dead-stimulus guard.
    assert rail_uni[0] == 0, f"the unipolar accumulator went negative, to {rail_uni[0]}"

    wide = ADD_SCALE_DYN["intwidth"] + 1
    wide_uni, _, _, _ = run("unipolar", segments_uni, plan, intwidth=wide)
    wide_bi, _, _, _ = run("bipolar", segments_bi, plan, intwidth=wide)
    assert wide_uni != out_uni, \
        "the unipolar outputs match a one-bit-wider accumulator, so WIDTH is unobservable"
    assert wide_bi != out_bi, \
        "the bipolar outputs match a one-bit-wider accumulator, so WIDTH is unobservable"

    # i_scale is an integer port, so only an integer scale has a hardware form, and
    # the model would quantize a non-integer request instead of rejecting it.
    for segment_index, schedule in enumerate(plan):
        for scale in schedule:
            assert float(scale).is_integer(), \
                f"i_scale carries whole units, so every requested scale must be an " \
                f"integer; segment {segment_index} requested {scale}"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALE_W {SCALE_W}\n"
        f"`define GEN_WIDTH {ADD_SCALE_DYN['intwidth']}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_PP_DELAY {pp_delay_uni}\n"
    )

    with VEC.open("w") as output:
        for segment_index, (segment_uni, segment_bi) in enumerate(
            zip(segments_uni, segments_bi)
        ):
            schedule = plan[segment_index]
            for cycle, (spikes_uni, spikes_bi) in enumerate(
                zip(segment_uni, segment_bi)
            ):
                reset = int(cycle == 0)
                scale_bits = format(schedule[cycle], f"0{SCALE_W}b")
                output.write(
                    f"{reset} {scale_bits} "
                    f"{bus(spikes_uni)} {out_uni[segment_index][cycle]} "
                    f"{bus(spikes_bi)} {out_bi[segment_index][cycle]}\n"
                )

    vector_count = sum(len(segment) for segment in segments_uni)
    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments_uni)} reset segments) "
        f"and {PARAMS} (SCALE_W={SCALE_W} WIDTH={ADD_SCALE_DYN['intwidth']} "
        f"ENTRY={ENTRY} PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
