import math
from pathlib import Path

import torch

from napl.sim.operation import div_scale_dyn, encode


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "div_scale_dyn.vec"
PARAMS = ROOT / "vec" / "div_scale_dyn_params.vh"

# Integer-grid carry regime (fracwidth 0, integer scales): the test's scale_max 4
# at intwidth 8 can never clamp, so this gen uses the narrow scale_max 3 at intwidth
# 3 that module-layer rule 9 licenses, where the bipolar clamp lands inside a vector
# row and main() proves WIDTH observable against a one-bit-wider model.
TIMESTEP = 256
DIV_SCALE_DYN = {"scale_max": 3, "intwidth": 3, "fracwidth": 0}
# The rail assertions in main() read the accumulator in raw units and compare against
# acc_max in value units, which agree only on the integer grid.
assert DIV_SCALE_DYN["fracwidth"] == 0, "rail assertions need the integer grid, fracwidth 0"
SCALE_MAX = DIV_SCALE_DYN["scale_max"]
# Bus width holding a scale in [1, scale_max].
SCALE_W = math.ceil(math.log2(SCALE_MAX + 1))
# One independent scalar circuit per test row: div_scale_dyn reduces nothing, so
# each of the 64 values is its own stream, spanning the test's range while keeping
# the vector file the same length.
SEGMENTS = 64


def test_values(polarity):
    """Return the test rows in the requested, probability-equivalent polarity."""
    low = 0.0 if polarity == "unipolar" else -1.0
    return torch.linspace(low, 1.0, SEGMENTS)


def scale_schedule(segment_index):
    """Return one segment's per-timestep scales, cycling the whole legal range.

    The last segment holds the largest scale, which is what drives the bipolar
    accumulator onto its clamp on that segment's all-ones stream.
    """
    if segment_index == SEGMENTS - 1:
        return [SCALE_MAX] * TIMESTEP
    return [1 + ((cycle + segment_index) % SCALE_MAX) for cycle in range(TIMESTEP)]


def schedules(segments):
    """Return the per-timestep scale plan of every segment, built once.

    run() and the integer check both read this same plan, so the check validates
    the schedule the outputs were generated from, not a freshly re-derived one.
    """
    return [scale_schedule(index) for index in range(len(segments))]


def encode_segments(polarity, values):
    """Encode each test value as one independent scalar RTL circuit's input stream."""
    segments = []
    for value in values:
        enc = encode({
            "polarity": polarity,
            "timestep": TIMESTEP,
            "generator": "sobol",
            "dim": 1,
        })
        enc.reset()
        source = value.reshape(1)
        segments.append([enc(source).clone() for _ in range(TIMESTEP)])
    return segments


def run(polarity, segments, plan, intwidth=None):
    """Generate bit-exact Python outputs and reset before every independent row.

    The largest pre-clamp sum reached over the whole run is returned with the
    outputs, so the caller can require the clamp to have been reached (bipolar) or
    to have stayed unreachable (unipolar).
    """
    config = dict(DIV_SCALE_DYN)
    if intwidth is not None:
        config["intwidth"] = intwidth
    model = div_scale_dyn({"polarity": polarity, **config})
    outputs = []
    peak = 0.0
    for segment, schedule in zip(segments, plan):
        model.reset()
        row = []
        for spikes, scale in zip(segment, schedule):
            offset = (1 - scale) / 2 if polarity == "bipolar" else 0.0
            prior = float(model.accumulator.item())
            peak = max(peak, prior + float(spikes.item()) - offset)
            row.append(int(model(spikes, scale).item()))
        outputs.append(row)
    return outputs, model.hw.pp_delay, peak, model


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar"))
    segments_bi = encode_segments("bipolar", test_values("bipolar"))
    assert len(segments_uni) == len(segments_bi) == SEGMENTS
    plan = schedules(segments_uni)
    out_uni, pp_delay_uni, peak_uni, model_uni = run("unipolar", segments_uni, plan)
    out_bi, pp_delay_bi, peak_bi, model_bi = run("bipolar", segments_bi, plan)
    assert pp_delay_uni == pp_delay_bi

    # Invariant check, not a stimulus check: the unipolar offset is 0, so every
    # addend is non-negative and a carry subtracts the current scale, confining the
    # pre-clamp sum to scale_max so the clamp is unreachable (which is why
    # div_scale_dyn_unipolar carries no WIDTH parameter) and this is the runnable
    # form of that induction.
    assert peak_uni <= model_uni.acc_max, \
        f"the unipolar pre-clamp sum reached {peak_uni}, past acc_max {model_uni.acc_max}"

    # Stimulus check: WIDTH is observable only through the bipolar clamp, so the run
    # must both reach it and produce outputs a wider accumulator would not.
    assert peak_bi > model_bi.acc_max, \
        f"the bipolar pre-clamp sum peaked at {peak_bi}, short of acc_max {model_bi.acc_max}"
    wide_bi, _, _, _ = run("bipolar", segments_bi, plan, intwidth=DIV_SCALE_DYN["intwidth"] + 1)
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
        f"`define GEN_WIDTH {DIV_SCALE_DYN['intwidth']}\n"
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
                    f"{int(spikes_uni.item())} {out_uni[segment_index][cycle]} "
                    f"{int(spikes_bi.item())} {out_bi[segment_index][cycle]}\n"
                )

    vector_count = sum(len(segment) for segment in segments_uni)
    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments_uni)} reset segments) "
        f"and {PARAMS} (SCALE_W={SCALE_W} WIDTH={DIV_SCALE_DYN['intwidth']} "
        f"PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
