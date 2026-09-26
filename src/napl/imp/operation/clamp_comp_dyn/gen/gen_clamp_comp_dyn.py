"""
Generate golden test vectors for the clamp_comp_dyn RTL straight from napl's
functional Python model (napl.sim.operation.clamp_comp_dyn) -- so the testbench
checks the Verilog against the *actual* simulator, not a hand-derived truth table.

clamp_comp_dyn chains two stream comparators: a maximum selector places the floor
and a minimum selector places the ceiling on its result. The bounds are streams the
caller supplies, so the RTL instantiates one max and one min cell, no encoder, and
no parameter. One bare module serves both polarities: the selectors compare spike
counts and never read the polarity. check_polarity_agreement() runs both polarity
models over the same spikes and requires them to agree, which is that single-module
claim in runnable form.

The stimulus mirrors tests/operation/test_clamp_comp_dyn.py: every value and bound
is a point of a GRID_LEN grid, with bipolar entries on even multiples of one
threshold step and unipolar entries on odd ones, so no unipolar stream repeats a
bipolar one. check_streams_distinct() requires that separation on the emitted
streams rather than on the requested rates.

Segments cover the legal value sweep against a static band, two inverted bands
(wide and one grid step narrow), and mid-run band changes driven with no reset
between the two bands.

Output: ../vec/clamp_comp_dyn.vec, one line per cycle:

    <rst> <i_input> <i_lo> <i_hi> <o_output>   (each 0/1, space-separated)

`rst`=1 marks the first cycle of each segment (the RTL testbench pulses i_rst_n low
there, returning to the exact post-reset() state). Each segment runs TIMESTEP
cycles, the full period of the Sobol encoders that draw the stimulus, so no encoder
index goes undriven.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_clamp_comp_dyn.py
"""
from pathlib import Path

import torch

from napl.sim.operation import clamp_comp_dyn, encode
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "clamp_comp_dyn.vec"
PARAMS = ROOT / "vec" / "clamp_comp_dyn_params.vh"

# Stimulus settings mirror tests/operation/test_clamp_comp_dyn.py.
TIMESTEP = 512
GRID_LEN = 256
INPUT_DIM, LO_DIM, HI_DIM = 1, 2, 3
# Static band per polarity, as grid step indices, both bounds inside the polarity's
# legal value range.
BAND_STEPS = {"bipolar": (32, 96), "unipolar": (16, 88)}
# The mid-run change drives the second band and then the third, with no reset.
SWEEP_STEPS = {
    "bipolar": ((32, 64), (80, 96)),
    "unipolar": ((24, 56), (72, 88)),
}
CHANGE_CYCLE = TIMESTEP // 2
# Inverted bands, as grid step indices: the wide one reverses a sweep band and the
# narrow one is a single grid step wide.
INVERTED_STEPS = {
    "bipolar": ((112, 16), (48, 47)),
    "unipolar": ((120, 8), (48, 47)),
}
# Bipolar sweep steps run rail to rail, so the first segment is the all-zero stream
# and the last the all-one stream; the unipolar grid sits off both rails by
# construction, so its sweep spans the range those odd multiples reach.
SWEEP_LEN = 33
INVERTED_VALUE_STEPS = (4, 64, 124)
CHANGE_VALUE_STEPS = (24, 64, 104)


def grid_value(polarity, step):
    """Return the grid value of one step index, in the polarity's own value range."""
    # A bipolar value v and the unipolar rate (v + 1) / 2 threshold against the same
    # sequence, so giving the two polarities opposite parities keeps every stream one
    # polarity drives off every stream the other drives.
    if polarity == "bipolar":
        return 2.0 * (2.0 * step / GRID_LEN) - 1.0
    return (2.0 * step + 1.0) / GRID_LEN


def sweep_steps(polarity):
    """Return the sweep step indices spanning the polarity's legal value range."""
    top = GRID_LEN // 2 if polarity == "bipolar" else GRID_LEN // 2 - 1
    return [round(index * top / (SWEEP_LEN - 1)) for index in range(SWEEP_LEN)]


def spike_streams(polarity, value_steps, band_steps):
    """Encode one per-cycle value and band, each on its own Sobol dimension."""
    encoders = [
        encode({"polarity": polarity, "timestep": TIMESTEP,
                "generator": "sobol", "dim": dim})
        for dim in (INPUT_DIM, LO_DIM, HI_DIM)
    ]
    for encoder in encoders:
        encoder.reset()
    streams = []
    for cycle in range(TIMESTEP):
        steps = (value_steps[cycle],) + band_steps[cycle]
        streams.append(tuple(
            int(encoder(torch.tensor(grid_value(polarity, step))).item())
            for encoder, step in zip(encoders, steps)
        ))
    return streams


def static_streams(polarity, value_step, band):
    """Encode one value against a band held for the whole segment."""
    return spike_streams(polarity, [value_step] * TIMESTEP, [band] * TIMESTEP)


def run(model, streams):
    """Drive the model from reset over one segment; return its per-cycle output."""
    model.reset()
    return [
        int(model(torch.tensor(float(spike)),
                  torch.tensor(float(lo)),
                  torch.tensor(float(hi))).item())
        for spike, lo, hi in streams
    ]


def sweep_segments(polarity, model):
    """Return (streams, output) per swept value against the static band."""
    band = BAND_STEPS[polarity]
    return [(streams, run(model, streams))
            for streams in (static_streams(polarity, step, band)
                            for step in sweep_steps(polarity))]


def inverted_segments(polarity, model):
    """Return (streams, output) per value against each inverted band."""
    rows = []
    for band in INVERTED_STEPS[polarity]:
        for step in INVERTED_VALUE_STEPS:
            streams = static_streams(polarity, step, band)
            rows.append((streams, run(model, streams)))
    return rows


def change_segments(polarity, model):
    """Return (streams, output) per value across a mid-run band change."""
    first, second = SWEEP_STEPS[polarity]
    bands = [first if cycle < CHANGE_CYCLE else second for cycle in range(TIMESTEP)]
    rows = []
    for step in CHANGE_VALUE_STEPS:
        streams = spike_streams(polarity, [step] * TIMESTEP, bands)
        rows.append((streams, run(model, streams)))
    return rows


def check_band(rows):
    """Require the two bipolar rails to place the band the bound streams carry.

    The bipolar sweep runs rail to rail, so its first segment is the all-zero stream
    and its last the all-one stream. On the first the floor selector holds on the lo
    stream every timestep and on the last the ceiling selector holds on the hi
    stream, so each output is that bound stream itself, cycle for cycle. A selector
    that stopped placing its bound departs from it.
    """
    for name, (streams, output), column, spike in (
        ("floor", rows[0], 1, 0), ("ceiling", rows[-1], 2, 1)
    ):
        assert all(row[0] == spike for row in streams), \
            f"the {name} segment is not the all-{spike} stream"
        bound = [row[column] for row in streams]
        departure = next((cycle for cycle, (out, want) in enumerate(zip(output, bound))
                          if out != want), None)
        assert departure is None, \
            f"the {name} selector emitted {sum(output)} spikes against a bound stream " \
            f"carrying {sum(bound)}, departing from it at cycle {departure}"


def check_inverted(polarity, rows):
    """Require an inverted band to hold the output between its two bound streams.

    The ceiling wins and the floor is ignored, so the output rate follows the hi
    rate rather than the input's. Each band's segments are checked to keep the
    output count inside the interval the two bound counts span, and at least one
    segment per band drives an input count outside that interval, so the property
    is not one an input passing straight through would also satisfy.
    """
    for band in range(0, len(rows), len(INVERTED_VALUE_STEPS)):
        outside = 0
        for streams, output in rows[band:band + len(INVERTED_VALUE_STEPS)]:
            count = sum(output)
            lo_count = sum(row[1] for row in streams)
            hi_count = sum(row[2] for row in streams)
            input_count = sum(row[0] for row in streams)
            assert hi_count < lo_count, \
                f"{polarity} band carries lo {lo_count} below hi {hi_count}, not inverted"
            assert hi_count <= count <= lo_count, \
                f"{polarity} inverted band emitted {count} spikes, outside the " \
                f"[{hi_count}, {lo_count}] its bound streams span"
            outside += not hi_count <= input_count <= lo_count
        assert outside, \
            f"{polarity} inverted band drove no input outside its bound interval"


def check_change(polarity, rows):
    """Require the mid-run band change to move the output rate with the band.

    Each half of the segment carries its own band, so the rate the band places on
    the input is the input count clamped into that half's bound counts. The two
    halves place different rates here, and the measured output must move the same
    way; a circuit that latched the first band would leave the halves equal.
    """
    for streams, output in rows:
        halves = []
        for start, stop in ((0, CHANGE_CYCLE), (CHANGE_CYCLE, TIMESTEP)):
            window = streams[start:stop]
            counts = [sum(row[column] for row in window) for column in range(3)]
            halves.append((sum(output[start:stop]),
                           min(max(counts[0], counts[1]), counts[2])))
        (before, first), (after, second) = halves
        assert second != first, \
            f"{polarity} band change places the same rate {first} on both halves"
        assert (after - before) * (second - first) > 0, \
            f"{polarity} band change moved the output from {before} to {after} " \
            f"spikes while the band moved the placed rate from {first} to {second}"


def check_polarity_agreement(rows):
    """Require both polarity models to agree on identical spikes.

    The RTL is one module for both polarities, which holds exactly because the
    polarity is stream metadata the selectors never read. Replaying the bipolar
    segments through the unipolar model is that claim as a run.
    """
    unipolar = clamp_comp_dyn({"polarity": "unipolar"})
    for streams, output in rows:
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
        binding = translate_node({"class": "clamp_comp_dyn",
                                  "config": {"polarity": polarity}})
        assert binding.rtl_module == "clamp_comp_dyn", \
            f"mapping resolves clamp_comp_dyn to {binding.rtl_module}"
        assert binding.parameters == {}, binding.parameters
        assert binding.port_map["inputs"] == {
            "input": "i_input", "lo": "i_lo", "hi": "i_hi"}, binding.port_map


def main():
    blocks = []
    bipolar_sweep = None
    for polarity in ("bipolar", "unipolar"):
        model = clamp_comp_dyn({"polarity": polarity})
        sweep = sweep_segments(polarity, model)
        if polarity == "bipolar":
            check_band(sweep)
            bipolar_sweep = sweep
        inverted = inverted_segments(polarity, model)
        check_inverted(polarity, inverted)
        changing = change_segments(polarity, model)
        check_change(polarity, changing)
        blocks += [(f"{polarity} sweep", sweep),
                   (f"{polarity} inverted", inverted),
                   (f"{polarity} band change", changing)]

    check_polarity_agreement(bipolar_sweep)
    distinct = check_streams_distinct(blocks)
    check_mapping()

    model = clamp_comp_dyn({"polarity": "bipolar"})
    VEC.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with VEC.open("w") as output:
        for _, rows in blocks:
            for streams, spikes in rows:
                for cycle, ((spike, lo, hi), out) in enumerate(zip(streams, spikes)):
                    output.write(f"{int(cycle == 0)} {spike} {lo} {hi} {out}\n")
                    count += 1

    PARAMS.write_text(
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
        f"`define GEN_VECTORS {count}\n"
    )
    print(f"wrote {VEC} ({count} vectors over {distinct} distinct segments) and "
          f"{PARAMS} (PP_DELAY={model.hw.pp_delay})")


if __name__ == "__main__":
    main()
