from pathlib import Path

import torch

from napl.sim.operation import div_scale, encode


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "div_scale.vec"
PARAMS = ROOT / "vec" / "div_scale_params.vh"

# Scale 3 at intwidth 3 is the narrow configuration module-layer rule 9 licenses on
# the integer grid (fracwidth 0), where the bipolar clamp lands inside a vector row and
# makes WIDTH observable, unlike the test's non-clamping scale 2 at intwidth 12.
TIMESTEP = 256
DIV_SCALE = {"scale": 3, "intwidth": 3, "fracwidth": 0}
# The rail assertions in main() read the accumulator in raw units and compare against
# acc_max in value units, which agree only on the integer grid.
assert DIV_SCALE["fracwidth"] == 0, "rail assertions need the integer grid, fracwidth 0"
# One independent scalar circuit per test row, matching the length of the test's
# 512-point make_values sweep since div_scale reduces nothing.
SEGMENTS = 64


def test_values(polarity):
    """Return the test rows in the requested, probability-equivalent polarity.

    The top row is the polarity's high rail, whose all-ones stream is what drives
    the bipolar accumulator onto its clamp.
    """
    low = 0.0 if polarity == "unipolar" else -1.0
    return torch.linspace(low, 1.0, SEGMENTS)


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


def run(polarity, segments, intwidth=None):
    """Generate bit-exact Python outputs and reset before every independent row.

    The largest pre-clamp sum reached over the whole run is returned with the
    outputs, so the caller can require the clamp to have been reached (bipolar) or
    to have stayed unreachable (unipolar).
    """
    config = dict(DIV_SCALE)
    if intwidth is not None:
        config["intwidth"] = intwidth
    model = div_scale({"polarity": polarity, **config})
    outputs = []
    peak = 0.0
    for segment in segments:
        model.reset()
        row = []
        for spikes in segment:
            prior = float(model.accumulator.item())
            peak = max(peak, prior + float(spikes.item()) - model.offset)
            row.append(int(model(spikes).item()))
        outputs.append(row)
    return outputs, model.hw.pp_delay, peak, model


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar"))
    segments_bi = encode_segments("bipolar", test_values("bipolar"))
    assert len(segments_uni) == len(segments_bi)
    out_uni, pp_delay_uni, peak_uni, model_uni = run("unipolar", segments_uni)
    out_bi, pp_delay_bi, peak_bi, model_bi = run("bipolar", segments_bi)
    assert pp_delay_uni == pp_delay_bi

    # Invariant check, not a stimulus check: the unipolar clamp is unreachable because
    # the offset of 0 confines the pre-clamp sum to scale, which is why
    # div_scale_unipolar carries no WIDTH parameter.
    assert peak_uni <= model_uni.acc_max, \
        f"the unipolar pre-clamp sum reached {peak_uni}, past acc_max {model_uni.acc_max}"

    # Stimulus check: WIDTH is observable only through the bipolar clamp, so the run
    # must both reach it and produce outputs a wider accumulator would not.
    assert peak_bi > model_bi.acc_max, \
        f"the bipolar pre-clamp sum peaked at {peak_bi}, short of acc_max {model_bi.acc_max}"
    wide_bi, _, _, _ = run("bipolar", segments_bi, intwidth=DIV_SCALE["intwidth"] + 1)
    assert wide_bi != out_bi, \
        "the bipolar outputs match a one-bit-wider accumulator, so WIDTH is unobservable"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALE {DIV_SCALE['scale']}\n"
        f"`define GEN_WIDTH {DIV_SCALE['intwidth']}\n"
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
                    f"{reset} {int(spikes_uni.item())} {out_uni[segment_index][cycle]} "
                    f"{int(spikes_bi.item())} {out_bi[segment_index][cycle]}\n"
                )

    vector_count = sum(len(segment) for segment in segments_uni)
    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments_uni)} reset segments) "
        f"and {PARAMS} (SCALE={DIV_SCALE['scale']} WIDTH={DIV_SCALE['intwidth']} "
        f"PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
