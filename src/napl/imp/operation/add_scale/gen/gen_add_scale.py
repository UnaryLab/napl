from pathlib import Path

import torch

from napl.sim.operation import add_scale, encode


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_scale.vec"
PARAMS = ROOT / "vec" / "add_scale_params.vh"

# Carry regime of test_add_scale.py (scale 2, intwidth 8) over the 8-entry
# reduction on the integer grid (fracwidth 0), with scale below the entry count
# so the bipolar offset drives the accumulator down and intwidth 8 so both clamps
# are reachable within a 256-step segment.
TIMESTEP = 256
ADD_SCALE = {"scale": 2, "intwidth": 8, "fracwidth": 0}
ENTRY = 8
# The rail assertions in main() read the accumulator in raw units and compare against
# acc_max and acc_min in value units, which agree only on the integer grid.
assert ADD_SCALE["fracwidth"] == 0, "rail assertions need the integer grid, fracwidth 0"
# Saturation segment (low rail, then high rail): the low rail carries the bipolar
# accumulator onto -2**(intwidth-1) and the high rail charges both polarities onto
# 2**(intwidth-1) - 1, so the recharge makes a corrupted clamp visible.
RAIL_DRAIN = 80
RAIL_CHARGE = 110
# Narrow regime: entry at or below scale, where the RTL accumulator holds only
# [0, scale-1] unipolar and [0, 2*scale-1] in bipolar half units. The first charge
# cycle after the drain lands both polarities on that top value.
ADD_SCALE_N = {"scale": 5, "intwidth": 8, "fracwidth": 0}
ENTRY_N = 4
assert ENTRY_N <= ADD_SCALE_N["scale"], "the narrow regime needs entry <= scale"
# The narrow instances share GEN_WIDTH with the wide ones.
assert ADD_SCALE_N["intwidth"] == ADD_SCALE["intwidth"], "one GEN_WIDTH serves both regimes"
# Boundary: entry one above scale, the smallest entry the RTL sizes wide. Its rail
# segment carries the unipolar accumulator above scale-1 and the bipolar one below
# 0, so a narrow register at this entry loses both.
ADD_SCALE_B = {"scale": 4, "intwidth": 8, "fracwidth": 0}
ENTRY_B = ADD_SCALE_B["scale"] + 1
assert ADD_SCALE_B["intwidth"] == ADD_SCALE["intwidth"], "one GEN_WIDTH serves every regime"


def test_values(polarity, entry):
    """Return the test rows in the requested, probability-equivalent polarity."""
    known = torch.full((8, entry), 0.25)
    fidelity = torch.linspace(-0.75, 0.75, 64 * entry).reshape(64, entry)
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


def rail_segment(polarity, entry):
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
    return [enc(torch.full((entry,), value)).clone() for value in values]


def run(polarity, segments, config):
    """Generate bit-exact Python outputs and reset before every independent row.

    The accumulator extremes of the last segment, the saturation one, are
    returned with the outputs so the caller can require it to have reached the
    clamps its width sets.
    """
    model = add_scale({"polarity": polarity, **config})
    outputs = []
    extremes = (0, 0)
    for segment in segments:
        model.reset()
        row = []
        low = high = 0
        for spikes in segment:
            row.append(int(model(spikes, dim=-1).item()))
            value = model.accumulator.item()
            low, high = min(low, value), max(high, value)
        outputs.append(row)
        extremes = (low, high)
    return outputs, model.hw.pp_delay, extremes, model


def bus(spikes):
    """Format lane 0 as the Verilog vector's least-significant bit."""
    return "".join(str(int(bit)) for bit in reversed(spikes.tolist()))


def main():
    segments_uni = encode_segments("unipolar", test_values("unipolar", ENTRY))
    segments_bi = encode_segments("bipolar", test_values("bipolar", ENTRY))
    segments_uni_n = encode_segments("unipolar", test_values("unipolar", ENTRY_N))
    segments_bi_n = encode_segments("bipolar", test_values("bipolar", ENTRY_N))
    segments_uni_b = encode_segments("unipolar", test_values("unipolar", ENTRY_B))
    segments_bi_b = encode_segments("bipolar", test_values("bipolar", ENTRY_B))
    segments_uni.append(rail_segment("unipolar", ENTRY))
    segments_bi.append(rail_segment("bipolar", ENTRY))
    segments_uni_n.append(rail_segment("unipolar", ENTRY_N))
    segments_bi_n.append(rail_segment("bipolar", ENTRY_N))
    segments_uni_b.append(rail_segment("unipolar", ENTRY_B))
    segments_bi_b.append(rail_segment("bipolar", ENTRY_B))
    assert len(segments_uni) == len(segments_bi) == len(segments_uni_n) == len(segments_bi_n) \
        == len(segments_uni_b) == len(segments_bi_b)
    out_uni, pp_delay_uni, rail_uni, model_uni = run("unipolar", segments_uni, ADD_SCALE)
    out_bi, pp_delay_bi, rail_bi, model_bi = run("bipolar", segments_bi, ADD_SCALE)
    out_uni_n, pp_delay_uni_n, rail_uni_n, _ = run("unipolar", segments_uni_n, ADD_SCALE_N)
    out_bi_n, pp_delay_bi_n, rail_bi_n, _ = run("bipolar", segments_bi_n, ADD_SCALE_N)
    out_uni_b, pp_delay_uni_b, rail_uni_b, _ = run("unipolar", segments_uni_b, ADD_SCALE_B)
    out_bi_b, pp_delay_bi_b, rail_bi_b, _ = run("bipolar", segments_bi_b, ADD_SCALE_B)
    assert pp_delay_uni == pp_delay_bi == pp_delay_uni_n == pp_delay_bi_n \
        == pp_delay_uni_b == pp_delay_bi_b
    scale_b = ADD_SCALE_B["scale"]
    assert rail_uni_b[1] > scale_b - 1, \
        f"the boundary unipolar rail segment peaked at {rail_uni_b[1]}, inside [0, {scale_b - 1}]"
    assert rail_bi_b[0] < 0, \
        f"the boundary bipolar rail segment bottomed at {rail_bi_b[0]}, not below 0"
    # The narrow rail segment must reach the top of the narrow state range, so a
    # narrow accumulator one bit short loses it; the bipolar top is a half unit.
    scale_n = ADD_SCALE_N["scale"]
    assert rail_uni_n == (0, scale_n - 1), \
        f"the narrow unipolar rail segment spans {rail_uni_n}, not (0, {scale_n - 1})"
    assert rail_bi_n == (0, scale_n - 0.5), \
        f"the narrow bipolar rail segment spans {rail_bi_n}, not (0, {scale_n - 0.5})"
    # One carry is subtracted after the clamp, so a clamped accumulator reads back
    # at acc_max - scale; the negative clamp never fires, so it reads back exactly.
    assert rail_uni[1] == model_uni.acc_max - ADD_SCALE["scale"], \
        f"the unipolar rail segment peaked at {rail_uni[1]}, short of {model_uni.acc_max}"
    assert rail_bi[1] == model_bi.acc_max - ADD_SCALE["scale"], \
        f"the bipolar rail segment peaked at {rail_bi[1]}, short of {model_bi.acc_max}"
    assert rail_bi[0] == model_bi.acc_min, \
        f"the bipolar rail segment bottomed at {rail_bi[0]}, short of {model_bi.acc_min}"
    # Invariant check, not a stimulus check: the unipolar negative clamp is
    # unreachable (offset 0, so every addend is non-negative and a carry only
    # subtracts to 0), so this is the runnable form of the induction the
    # add_scale_unipolar header states, not a dead-stimulus guard.
    assert rail_uni[0] == 0, f"the unipolar accumulator went negative, to {rail_uni[0]}"

    VEC.parent.mkdir(parents=True, exist_ok=True)

    vector_count = 0
    with VEC.open("w") as output:
        for segment_index, segment in enumerate(
            zip(segments_uni, segments_bi, segments_uni_n, segments_bi_n,
                segments_uni_b, segments_bi_b)
        ):
            for cycle, (spikes_uni, spikes_bi, spikes_uni_n, spikes_bi_n,
                        spikes_uni_b, spikes_bi_b) in enumerate(zip(*segment)):
                reset = int(cycle == 0)
                output.write(
                    f"{reset} {bus(spikes_uni)} {out_uni[segment_index][cycle]} "
                    f"{bus(spikes_bi)} {out_bi[segment_index][cycle]} "
                    f"{bus(spikes_uni_n)} {out_uni_n[segment_index][cycle]} "
                    f"{bus(spikes_bi_n)} {out_bi_n[segment_index][cycle]} "
                    f"{bus(spikes_uni_b)} {out_uni_b[segment_index][cycle]} "
                    f"{bus(spikes_bi_b)} {out_bi_b[segment_index][cycle]}\n"
                )
                vector_count += 1

    PARAMS.write_text(
        f"`define GEN_SCALE {ADD_SCALE['scale']}\n"
        f"`define GEN_WIDTH {ADD_SCALE['intwidth']}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_SCALE_N {ADD_SCALE_N['scale']}\n"
        f"`define GEN_ENTRY_N {ENTRY_N}\n"
        f"`define GEN_SCALE_B {ADD_SCALE_B['scale']}\n"
        f"`define GEN_ENTRY_B {ENTRY_B}\n"
        f"`define GEN_PP_DELAY {pp_delay_uni}\n"
        f"`define GEN_VECTORS {vector_count}\n"
    )

    print(
        f"wrote {VEC} ({vector_count} vectors, {len(segments_uni)} reset segments) "
        f"and {PARAMS} (SCALE={ADD_SCALE['scale']} WIDTH={ADD_SCALE['intwidth']} "
        f"ENTRY={ENTRY}; narrow SCALE={ADD_SCALE_N['scale']} ENTRY={ENTRY_N}; "
        f"boundary SCALE={ADD_SCALE_B['scale']} ENTRY={ENTRY_B}; "
        f"PP_DELAY={pp_delay_uni})"
    )


if __name__ == "__main__":
    main()
