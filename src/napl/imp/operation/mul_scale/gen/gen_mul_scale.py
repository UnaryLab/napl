from pathlib import Path

import torch

from napl.sim.operation import mul_scale, encode
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "mul_scale.vec"
PARAMS = ROOT / "vec" / "mul_scale_params.vh"

# Scale 3 at intwidth 3 is the narrow configuration on the integer grid (fracwidth 0)
# where both clamps land inside a vector row and make WIDTH observable, unlike the
# test's non-clamping scale 2 at intwidth 12. mul_scale amplifies (scale > 1), so both
# the upper clamp (inflow outpaces the one-unit drain) and, for bipolar, the lower
# clamp (the negated offset drives the sum below zero) are reachable.
TIMESTEP = 256
MUL_SCALE = {"scale": 3, "intwidth": 3, "fracwidth": 0}
# The rail assertions in main() read the accumulator in raw units and compare against
# acc_max/acc_min in value units, which agree only on the integer grid.
assert MUL_SCALE["fracwidth"] == 0, "rail assertions need the integer grid, fracwidth 0"
# One independent scalar circuit per test row; mul_scale reduces nothing.
SEGMENTS = 64


def test_values(polarity):
    """Return the test rows narrowed so x * scale stays a legal rate, as the test does.

    Unipolar inputs span [0, 1/scale] and bipolar [-1/scale, 1/scale], whose products
    reach the full [0, 1] and [-1, 1] output rails.
    """
    scale = MUL_SCALE["scale"]
    high = 1.0 / scale
    low = 0.0 if polarity == "unipolar" else -high
    return torch.linspace(low, high, SEGMENTS)


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

    The largest and smallest pre-clamp sums reached over the whole run are returned
    with the outputs, so the caller can require a clamp to have been reached.
    """
    config = dict(MUL_SCALE)
    if intwidth is not None:
        config["intwidth"] = intwidth
    model = mul_scale({"polarity": polarity, **config})
    outputs = []
    peak_hi = -1e18
    peak_lo = 1e18
    for segment in segments:
        model.reset()
        row = []
        for spikes in segment:
            prior = float(model.accumulator.item())
            pre = prior + float(spikes.item()) * model.scale_raw - model.offset
            peak_hi = max(peak_hi, pre)
            peak_lo = min(peak_lo, pre)
            row.append(int(model(spikes).item()))
        outputs.append(row)
    return outputs, model.hw.pp_delay, peak_hi, peak_lo, model


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar"))
    segments_bi = encode_segments("bipolar", test_values("bipolar"))
    assert len(segments_uni) == len(segments_bi)
    out_uni, pp_delay_uni, hi_uni, lo_uni, model_uni = run("unipolar", segments_uni)
    out_bi, pp_delay_bi, hi_bi, lo_bi, model_bi = run("bipolar", segments_bi)
    assert pp_delay_uni == pp_delay_bi

    # Stimulus check: WIDTH is observable only through a clamp, so the run must reach
    # one and produce outputs a wider accumulator would not. The unipolar upper clamp
    # engages because scale > 1 lets the inflow outpace the one-unit drain.
    assert hi_uni > model_uni.acc_max, \
        f"the unipolar pre-clamp sum peaked at {hi_uni}, short of acc_max {model_uni.acc_max}"
    wide_uni, _, _, _, _ = run("unipolar", segments_uni, intwidth=MUL_SCALE["intwidth"] + 1)
    assert wide_uni != out_uni, \
        "the unipolar outputs match a one-bit-wider accumulator, so WIDTH is unobservable"

    # The bipolar negated offset drives the sum below zero, so the lower clamp engages
    # too; requiring both rails exercises the signed datapath's clamp arms.
    assert hi_bi > model_bi.acc_max, \
        f"the bipolar pre-clamp sum peaked at {hi_bi}, short of acc_max {model_bi.acc_max}"
    assert lo_bi < model_bi.acc_min, \
        f"the bipolar pre-clamp sum bottomed at {lo_bi}, above acc_min {model_bi.acc_min}"
    wide_bi, _, _, _, _ = run("bipolar", segments_bi, intwidth=MUL_SCALE["intwidth"] + 1)
    assert wide_bi != out_bi, \
        "the bipolar outputs match a one-bit-wider accumulator, so WIDTH is unobservable"

    # Resolve each elaborated variant's mapping entry and require it to reproduce
    # the params these vectors were built with, so make test gates the mapping too.
    for polarity in ("unipolar", "bipolar"):
        binding = translate_node({"class": "mul_scale", "config": {
            "polarity": polarity, "scale": MUL_SCALE["scale"],
            "intwidth": MUL_SCALE["intwidth"], "fracwidth": MUL_SCALE["fracwidth"]}})
        expected = {"SCALE": MUL_SCALE["scale"], "WIDTH": MUL_SCALE["intwidth"]}
        assert binding.parameters == expected, \
            f"mapping mul_scale_{polarity} resolves {binding.parameters}, not {expected}"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALE {MUL_SCALE['scale']}\n"
        f"`define GEN_WIDTH {MUL_SCALE['intwidth']}\n"
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
        f"and {PARAMS} (SCALE={MUL_SCALE['scale']} WIDTH={MUL_SCALE['intwidth']} "
        f"PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
