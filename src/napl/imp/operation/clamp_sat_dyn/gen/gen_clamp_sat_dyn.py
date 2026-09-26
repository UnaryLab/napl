"""
Generate golden test vectors for the clamp_sat_dyn RTL straight from napl's functional
Python model (napl.sim.operation.clamp_sat_dyn) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

clamp_sat_dyn chains three unit-scale bipolar add_scale stages whose saturations place
the band the two bound streams carry. The bounds are streams the caller supplies,
so the RTL instantiates three add_scale_bipolar cells and no encoder, and one
module serves both polarities: the same spikes read as the bipolar value 2p - 1
for the input and for both bounds alike. main() runs both polarity models over the
same spikes and requires them to agree, which is that single-module claim in
runnable form.

The two polarity halves carry disjoint stimulus. A rate is all a Sobol encoder
reads, so a bipolar band and sweep whose rates match a unipolar band and sweep
encode to the same spikes and the second half would re-run the first. The bands
are chosen apart in rate, the unipolar sweep is built from the rates whose stream
no bipolar sweep segment already carries, and check_streams_distinct() requires
that separation on the emitted streams rather than on the requested rates.

Output: ../vec/clamp_sat_dyn.vec, one line per cycle:

    <rst> <i_input> <i_lo> <i_hi> <o_output>   (each 0/1, space-separated)

`rst`=1 marks the first cycle of each independent segment (the RTL testbench pulses
i_rst_n low there, returning to the exact post-reset() state). Each segment runs
TIMESTEP cycles, the full period of the Sobol encoders that draw the stimulus, so
no encoder index goes undriven.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_clamp_sat_dyn.py
"""
import random
from pathlib import Path

import torch

from napl.sim.operation import clamp_sat_dyn, encode
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "clamp_sat_dyn.vec"
PARAMS = ROOT / "vec" / "clamp_sat_dyn_params.vh"

# Stimulus settings mirror tests/operation/test_clamp_sat_dyn.py: a band inside each
# polarity's legal value range, the input and the two bound streams on their own
# Sobol dimensions, and the full legal value sweep so values below lo and above hi
# both drive the stages.
TIMESTEP = 256
INPUT_DIM, LO_DIM, HI_DIM = 1, 2, 3
# The four bound rates are (0.125, 0.625) bipolar against (0.25, 0.75) unipolar, so
# neither bound stream of one polarity repeats a bound stream of the other.
BOUNDS = {"bipolar": (-0.75, 0.25), "unipolar": (0.25, 0.75)}
BIPOLAR_SWEEP = torch.linspace(-1.0, 1.0, 128)
# An inverted band, where the lo stream decodes above the hi stream, is a property
# of the streams and not of the configuration, so the RTL must carry it: the model
# emits the hi stream itself there. A short value list suffices because the claim
# is that the input is ignored, and these values sit below, inside, and above the
# reversed bounds.
INVERTED_BOUNDS = {"bipolar": (0.25, -0.75), "unipolar": (0.75, 0.25)}
BIPOLAR_INVERTED_SWEEP = torch.tensor([-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0])
INVERTED_SEGMENTS = len(BIPOLAR_INVERTED_SWEEP)
# A Sobol encoder reads its value only through the 1/TIMESTEP threshold grid, so two
# rates encode to the same stream unless a grid point separates them. Candidate
# unipolar rates therefore sit midway between grid points, and the ones whose stream
# a bipolar segment already carries are dropped.
CANDIDATE_RATES = [(n + 0.5) / TIMESTEP for n in range(TIMESTEP)]

# Rails of a stage accumulator at intwidth 4 and fracwidth 0, restated here because
# floor_rail_output() derives the floor rail from the emission rule alone; main()
# requires them to match the model's stages.
ACC_MIN, ACC_MAX = -8, 7
# Random rails that check floor_rail_output() against the model, over the rate range
# the bound encoders cover.
PROVENANCE_RAILS = 200
PROVENANCE_RATES = (0.05, 0.9)

# Band changes inside one segment: the band moves at CHANGE_CYCLE while the input
# rate is held, so the segment cosimulates the settling the class documents rather
# than a second static band. The listed direction is the sign the output rate moves
# in, which is what the transient has to carry: the first half sits on the first
# band and the second half leaves it.
CHANGE_CYCLE = TIMESTEP // 2
BAND_CHANGE = [
    # (label, input value, first band, second band, output-rate direction)
    ("widening", 0.6, (-0.25, 0.25), (-1.0, 1.0), +1),
    ("narrowing", 0.8, (-0.9, 0.9), (-0.2, 0.2), -1),
    ("bounds crossing the input", 0.4, (0.75, 1.0), (-1.0, 0.0), -1),
    ("inverted band becoming a band", 0.0, (1.0, -1.0), (-0.5, 0.5), +1),
]


def spike_streams(polarity, value, bands):
    """Encode one value and a per-cycle band, each on its own Sobol dimension."""
    encoders = [
        encode({"polarity": polarity, "timestep": TIMESTEP,
                "generator": "sobol", "dim": dim})
        for dim in (INPUT_DIM, LO_DIM, HI_DIM)
    ]
    for encoder in encoders:
        encoder.reset()
    return [
        tuple(int(encoder(torch.tensor(float(item))).item())
              for encoder, item in zip(encoders, (value,) + bands[cycle]))
        for cycle in range(TIMESTEP)
    ]


def static_streams(polarity, value, bounds):
    """Encode one value against a band that is held for the whole segment."""
    return spike_streams(polarity, value, [bounds] * TIMESTEP)


def input_stream(polarity, value):
    """Return the input-dimension stream one value encodes to."""
    return tuple(spike for spike, _, _ in static_streams(polarity, value, (0.0, 0.0)))


def run(model, streams):
    """Drive the model from reset over one segment; return its per-cycle output."""
    model.reset()
    return [
        int(model(torch.tensor(float(spike)),
                  torch.tensor(float(lo)),
                  torch.tensor(float(hi))).item())
        for spike, lo, hi in streams
    ]


def segments(polarity, bounds, sweep):
    """Return (stream, output) per value, and the model that produced them."""
    model = clamp_sat_dyn({"polarity": polarity})
    rows = []
    for value in sweep:
        streams = static_streams(polarity, value, bounds)
        rows.append((streams, run(model, streams)))
    return rows, model


def band_change_segments():
    """Return (stream, output) per band change, on the bipolar model."""
    model = clamp_sat_dyn({"polarity": "bipolar"})
    rows = []
    for label, value, first, second, direction in BAND_CHANGE:
        bands = [first if cycle < CHANGE_CYCLE else second for cycle in range(TIMESTEP)]
        streams = spike_streams("bipolar", value, bands)
        output = run(model, streams)
        check_band_change(label, output, direction)
        rows.append((streams, output))
    return rows


def unipolar_sweeps(taken):
    """Split the rates no bipolar segment carries into a sweep and an inverted sweep.

    The full sweep keeps the rails 0.0 and 1.0, whose streams the bipolar sweep also
    carries, because check_band() reads the all-zero and all-one segments to see the
    two saturations; every other segment of both sweeps is a rate the bipolar half
    does not encode.
    """
    free = [value for value in CANDIDATE_RATES if input_stream("unipolar", value) not in taken]
    assert len(free) > INVERTED_SEGMENTS, \
        f"only {len(free)} unipolar rates are free of the bipolar streams"
    step = len(free) // INVERTED_SEGMENTS
    inverted = [free[index * step] for index in range(INVERTED_SEGMENTS)]
    return (torch.tensor([0.0] + free + [1.0]),
            torch.tensor(inverted))


def floor_rail_output(lo_stream, hi_stream):
    """Return the floor rail's output, derived from the add_scale emission rule.

    On the all-zero input stream the floor stage's per-timestep delta, input - lo, is
    never positive, so that stage's accumulator only falls and the stage emits
    nothing. The band is placed by the two stages behind it. Each accumulates its
    delta, emits a spike once the accumulator reaches the unit scale, and drains that
    unit on emission: the ceiling stage carries lo + 1 - hi and the shift stage
    carries that carry plus hi - 1. While the ceiling accumulator stays off its rails
    a lo spike therefore reaches the output on the next timestep whose hi spike is
    high, which is the same timestep whenever hi is already high; a clamp at ACC_MAX
    drops backlog that no later hi spike carries out.
    """
    ceiling_acc = shift_acc = 0
    output = []
    for lo, hi in zip(lo_stream, hi_stream):
        ceiling_acc = max(ACC_MIN, min(ACC_MAX, ceiling_acc + lo + 1 - hi))
        carry = int(ceiling_acc >= 1)
        ceiling_acc -= carry
        shift_acc = max(ACC_MIN, min(ACC_MAX, shift_acc + carry + hi - 1))
        spike = int(shift_acc >= 1)
        shift_acc -= spike
        output.append(spike)
    return output


def check_floor_rail_form():
    """Require floor_rail_output() to reproduce the model on random bound rails.

    The closed form is a derivation from the emission rule rather than a recording of
    this file's stimulus, so it has to hold on rails the sweep never runs: random lo
    and hi streams over the rate range the bound encoders cover, against the all-zero
    input the floor rail carries.

    The rails also have to reach the clamping regime, since the emission rule holds
    unconditionally only while the ceiling accumulator stays off ACC_MAX: a sweep
    idling below the rail would agree with the model without exercising the clamp.
    """
    model = clamp_sat_dyn({"polarity": "bipolar"})
    # The seed pins the sweep, so its coverage is a fixed pair of numbers rather than a
    # sample: 102 of the 200 rails clamp, over 5824 clamp events.
    rng = random.Random(0)
    low, high = PROVENANCE_RATES
    clamps = 0
    for trial in range(PROVENANCE_RAILS):
        lo_rate, hi_rate = rng.uniform(low, high), rng.uniform(low, high)
        lo_stream = [int(rng.random() < lo_rate) for _ in range(TIMESTEP)]
        hi_stream = [int(rng.random() < hi_rate) for _ in range(TIMESTEP)]
        output = run(model, list(zip([0] * TIMESTEP, lo_stream, hi_stream)))
        expected = floor_rail_output(lo_stream, hi_stream)
        departure = next((cycle for cycle, (out, want) in enumerate(zip(output, expected))
                          if out != want), None)
        assert departure is None, \
            f"floor_rail_output departs from the model at cycle {departure} on random " \
            f"rail {trial} (lo rate {lo_rate:.3f}, hi rate {hi_rate:.3f})"
        ceiling_acc = 0
        for lo, hi in zip(lo_stream, hi_stream):
            ceiling_acc += lo + 1 - hi
            clamps += ceiling_acc > ACC_MAX
            ceiling_acc = max(ACC_MIN, min(ACC_MAX, ceiling_acc))
            ceiling_acc -= int(ceiling_acc >= 1)
    assert clamps, \
        f"no ceiling accumulator clamped over {PROVENANCE_RAILS} rails, so the sweep " \
        f"never left the regime where the emission rule holds unconditionally"


def check_band(polarity, rows):
    """Require the extreme segments to place the band the bound streams carry.

    The sweep runs from the bottom of the polarity's legal range to the top, so its
    first segment is the all-zero stream and its last the all-one stream. On the
    first the floor stage emits nothing and the output is floor_rail_output() cycle
    for cycle, the band placed by the ceiling and shift stages behind it. On the last
    the ceiling stage saturates every timestep and the output is the hi stream itself,
    cycle for cycle.
    A stage that stopped placing its bound departs from the sequence it owns.
    """
    hi_bound = BOUNDS[polarity][1]
    bipolar = {"bipolar": lambda value: value, "unipolar": lambda value: 2.0 * value - 1.0}[polarity]
    for name, (streams, output) in (("floor", rows[0]), ("ceiling", rows[-1])):
        expected_spike = 0 if name == "floor" else 1
        assert all(spike == expected_spike for spike, _, _ in streams), \
            f"{polarity} {name} segment is not the all-{expected_spike} stream"
        if name == "floor":
            expected = floor_rail_output([row[1] for row in streams],
                                         [row[2] for row in streams])
        else:
            expected = [row[2] for row in streams]
            count = sum(output)
            target = round(TIMESTEP * (bipolar(hi_bound) + 1.0) / 2.0)
            assert count == target, \
                f"{polarity} ceiling saturation emitted {count} spikes, not {target}"
        departure = next((cycle for cycle, (out, want) in enumerate(zip(output, expected))
                          if out != want), None)
        assert departure is None, \
            f"{polarity} {name} rail departs from the sequence it places at cycle {departure}"


def check_inverted(polarity, rows):
    """Require the inverted band to emit the hi stream, whatever the input is."""
    for streams, output in rows:
        expected = [hi for _, _, hi in streams]
        assert output == expected, \
            f"{polarity} inverted band did not emit the hi stream"


def check_band_change(label, output, direction):
    """Require the band change to move the output rate in the direction it should."""
    before = sum(output[:CHANGE_CYCLE])
    after = sum(output[CHANGE_CYCLE:])
    moved = (after - before) * direction
    assert moved > 0, \
        f"{label} band change moved the output rate from {before} to {after} spikes, " \
        f"not in direction {direction}"


def check_polarity_agreement(rows_by_polarity):
    """Require both polarity models to agree on identical spikes.

    The RTL is one module for both polarities, which holds exactly because the
    polarity is stream metadata the stages never read. Replaying the bipolar
    segments through the unipolar model is that claim as a run.
    """
    unipolar = clamp_sat_dyn({"polarity": "unipolar"})
    for streams, output in rows_by_polarity["bipolar"]:
        assert run(unipolar, streams) == output, \
            "the unipolar and bipolar models disagree on identical spikes"


def check_streams_distinct(blocks):
    """Require every segment to carry a stimulus no other segment already carries.

    A duplicate segment is a vector row that proves nothing the file already proved,
    and a rate is all a Sobol encoder reads, so two polarities asking for the same
    rates emit the same spikes. Comparing the emitted (input, lo, hi) streams catches
    that whatever the requested values were.
    """
    seen = {}
    for name, rows in blocks:
        for index, (streams, _) in enumerate(rows):
            key = tuple(streams)
            assert key not in seen, \
                f"{name} segment {index} repeats the stimulus of {seen[key]}"
            seen[key] = f"{name} segment {index}"
    return len(seen)


def check_mapping():
    """Require the mapping entry to resolve this one module for both polarities."""
    for polarity in ("unipolar", "bipolar"):
        binding = translate_node({"class": "clamp_sat_dyn", "config": {"polarity": polarity}})
        assert binding.rtl_module == "clamp_sat_dyn", \
            f"mapping resolves clamp_sat_dyn to {binding.rtl_module}"
        assert binding.port_map["inputs"] == {
            "input": "i_input", "lo": "i_lo", "hi": "i_hi"}, binding.port_map


def main():
    rows_by_polarity = {}
    inverted_by_polarity = {}

    check_floor_rail_form()

    rows_by_polarity["bipolar"], model = segments(
        "bipolar", BOUNDS["bipolar"], BIPOLAR_SWEEP)
    check_band("bipolar", rows_by_polarity["bipolar"])
    inverted_by_polarity["bipolar"], _ = segments(
        "bipolar", INVERTED_BOUNDS["bipolar"], BIPOLAR_INVERTED_SWEEP)
    check_inverted("bipolar", inverted_by_polarity["bipolar"])

    taken = {tuple(spike for spike, _, _ in streams)
             for rows in (rows_by_polarity["bipolar"], inverted_by_polarity["bipolar"])
             for streams, _ in rows}
    sweep, inverted_sweep = unipolar_sweeps(taken)
    rows_by_polarity["unipolar"], _ = segments("unipolar", BOUNDS["unipolar"], sweep)
    check_band("unipolar", rows_by_polarity["unipolar"])
    inverted_by_polarity["unipolar"], _ = segments(
        "unipolar", INVERTED_BOUNDS["unipolar"], inverted_sweep)
    check_inverted("unipolar", inverted_by_polarity["unipolar"])

    changing = band_change_segments()
    check_polarity_agreement(rows_by_polarity)
    check_mapping()

    blocks = [
        ("unipolar sweep", rows_by_polarity["unipolar"]),
        ("bipolar sweep", rows_by_polarity["bipolar"]),
        ("unipolar inverted", inverted_by_polarity["unipolar"]),
        ("bipolar inverted", inverted_by_polarity["bipolar"]),
        ("band change", changing),
    ]
    distinct = check_streams_distinct(blocks)

    # The RTL inherits the stage accumulator width from the model rather than
    # restating it, and the unit scale on the integer grid is what makes every
    # stage offset a whole accumulator unit, so a model that moved off either
    # would need a different circuit and fails here instead.
    stage = model.floor_stage
    assert stage.scale == 1 and stage.fracwidth == 0, \
        f"clamp_sat_dyn stages are scale {stage.scale} on a fracwidth {stage.fracwidth} grid"
    width = stage.intwidth
    assert (ACC_MIN, ACC_MAX) == (-2 ** (width - 1), 2 ** (width - 1) - 1), \
        f"floor_rail_output rails {(ACC_MIN, ACC_MAX)} are not the stage rails at intwidth {width}"
    for other in (model.ceiling_stage, model.shift_stage):
        assert other.intwidth == width, "the three stages carry different intwidths"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with VEC.open("w") as output:
        for _, rows in blocks:
            for streams, spikes in rows:
                for cycle, ((spike, lo, hi), out) in enumerate(zip(streams, spikes)):
                    output.write(f"{int(cycle == 0)} {spike} {lo} {hi} {out}\n")
                    count += 1

    PARAMS.write_text(
        f"`define GEN_WIDTH {width}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
        f"`define GEN_VECTORS {count}\n"
    )
    print(f"wrote {VEC} ({count} vectors over {distinct} distinct segments) and {PARAMS} "
          f"(WIDTH={width} PP_DELAY={model.hw.pp_delay})")


if __name__ == "__main__":
    main()
