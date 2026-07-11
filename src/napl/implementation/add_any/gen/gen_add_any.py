"""
Generate golden test vectors for the add_any RTL modules straight from napl's
functional Python model (napl.operation.add_any) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

add_any is a stateful per-timestep accumulator. Per timestep it adds the input
partial sum minus a constant offset, clamps to the width-sized accumulator range,
emits out = (acc >= scale), and subtracts scale where it fired. The bit-serial
RTL drives the pre-reduced partial sum directly (the model's dim=None path), so
each cycle's input is one integer in [0, ENTRY]. Polarity only changes the offset
(bipolar: (ENTRY-SCALE)/2; unipolar: 0), so each polarity is its own module.

Faithfulness to test_add_any.py: that test encodes an (rows, SCALE) input with
the codec (sobol, timestep=256), reduces over the lane dim, and feeds the result
to add_any. ENTRY = SCALE = the reduction dim. Here we reproduce the SAME
per-timestep partial-sum streams by encoding representative per-lane value vectors
with the test's exact encoder config and summing the lane spikes per timestep.
The streams therefore match the spike trains the test sends the op bit-for-bit.

Sizing is the single source of truth: SCALE/WIDTH (from add_any_config) and ENTRY
(= the reduction dim, == SCALE) are read once here, used to build the model, AND
emitted into ../vec/add_any_params.vh as `GEN_SCALE / `GEN_WIDTH / `GEN_ENTRY so
the testbench overrides the RTL parameters with the same values. RTL and sim
cannot drift.

Output: ../vec/add_any.vec, one line per cycle:

    <rst> <partial> <out_unipolar> <out_bipolar>   (rst,outs 0/1; partial decimal)

`rst`=1 marks a cycle where the model was reset() immediately before this forward()
(the accumulator is 0 entering this cycle); the testbench pulses i_rst_n low before
driving that row. The first row carries rst=1, and a MID-STREAM reset row also
carries rst=1 -- proving the RTL's active-low reset returns to the model's reset()
state from a dirtied accumulator.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_add_any.py
"""
from pathlib import Path

import torch
from napl.operation import add_any
from napl.module import encoder

VEC = Path(__file__).resolve().parent.parent / "vec" / "add_any.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "add_any_params.vh"

# test_add_any.py add_any_config: the sizing params the op is built with.
ADD_ANY = {"scale": 128, "width": 20}
# ENTRY = number of addends = the reduction dim = SCALE (test input shape (rows, scale)).
ENTRY = ADD_ANY["scale"]
SCALE = ADD_ANY["scale"]
WIDTH = ADD_ANY["width"]

# test_add_any.py codec_config: the encoder feeding add_any.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol"}


def encode_lane_stream(codec_config, values):
    """Per-timestep partial-sum stream for an ENTRY-long per-lane value vector.

    Drives a real napl encoder built from codec_config (the test's exact config,
    single sobol dim shared across lanes, just like the test) over `timestep`
    cycles from a fresh reset, summing the lane spikes each cycle. Returns a list
    of length timestep of integer partial sums in [0, ENTRY].
    """
    enc = encoder(dict(codec_config))
    enc.reset()
    v = torch.tensor([float(x) for x in values]).type(enc.num_seq.dtype)
    partials = []
    for _ in range(codec_config["timestep"]):
        spike = enc(v)                      # one spike per lane this timestep
        partials.append(int(spike.sum().item()))
    return partials


def lane_vectors(polarity, entry):
    """Representative ENTRY-long per-lane value vectors covering the accumulator's
    range: all-low / all-high rails, midpoints, graded fills, and random draws."""
    lo, hi = (-1.0, 1.0) if polarity == "bipolar" else (0.0, 1.0)
    mid = (lo + hi) / 2
    g = torch.Generator().manual_seed(20240613)
    vecs = []
    # rails and midpoint -> drive the accumulator toward both clamp bounds
    vecs.append([hi] * entry)               # max partial each cycle: positive rail
    vecs.append([lo] * entry)               # min partial each cycle: negative rail
    vecs.append([mid] * entry)              # midpoint: hovers near threshold
    # graded fills: a fraction of lanes high, rest low
    for frac in (0.25, 0.5, 0.75):
        k = int(entry * frac)
        vecs.append([hi] * k + [lo] * (entry - k))
    # deterministic random draws over the operand range
    for _ in range(3):
        vecs.append((lo + (hi - lo) * torch.rand(entry, generator=g)).tolist())
    return vecs


def run(polarity, segments):
    """Run the model over a list of partial-sum segments, resetting before the
    first segment and AGAIN before the last (mid-stream reset). Returns
    (rst_flags, outs): rst_flags[i]=1 on the first cycle after each reset()."""
    model = add_any(config={"polarity": polarity, "scale": SCALE, "width": WIDTH})
    model.reset()
    rsts, outs = [], []
    first = True
    for seg_idx, seg in enumerate(segments):
        if seg_idx == len(segments) - 1:
            model.reset()                   # mid-stream reset before the final segment
            first = True
        for j, p in enumerate(seg):
            inp = torch.tensor(p, dtype=model.ntype)
            entry = ENTRY if first else None
            outs.append(int(model(inp, entry=entry, dim=None).item()))
            # rst=1 on the very first cycle after each reset() (the dirtied-state proof)
            rsts.append(1 if first else 0)
            first = False
    return rsts, outs


def build_segments():
    # encode each representative lane vector into a partial-sum segment.
    # use the bipolar codec (the test's) for the stream shape; the partial sums
    # are polarity-agnostic integers in [0, ENTRY], so both models see the same
    # input stream and only the offset differs between the two modules.
    vecs = lane_vectors(CODEC["polarity"], ENTRY)
    return [encode_lane_stream(CODEC, v) for v in vecs]


def main():
    segments = build_segments()
    flat_partials = [p for seg in segments for p in seg]
    rst_uni, out_uni = run("unipolar", segments)
    rst_bi, out_bi = run("bipolar", segments)
    # the reset schedule is identical for both polarities (same segmentation).
    assert rst_uni == rst_bi

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameters.
    PARAMS.write_text(
        f"`define GEN_SCALE {SCALE}\n"
        f"`define GEN_WIDTH {WIDTH}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
    )

    with VEC.open("w") as f:
        for r, p, u, b in zip(rst_uni, flat_partials, out_uni, out_bi):
            f.write(f"{r} {p} {u} {b}\n")
    print(
        f"wrote {VEC} ({len(flat_partials)} vectors) and {PARAMS} "
        f"(GEN_SCALE={SCALE} GEN_WIDTH={WIDTH} GEN_ENTRY={ENTRY})"
    )


if __name__ == "__main__":
    main()
