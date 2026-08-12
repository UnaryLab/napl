"""
Generate golden test vectors for the clamp RTL straight from napl's functional
Python model (napl.sim.operation.clamp) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

clamp is a stateful per-timestep streaming op (a saturating counter over a
running acc/timestep estimate) with distinct unipolar and bipolar datapaths, so
there are two variants: clamp_unipolar and clamp_bipolar. Its sizing keys are the
fixed-point band [lo, hi] and its grid fracwidth; the RTL sizes its counters from
the run length, so TIMESTEP is emitted alongside. The band is chosen (mirroring
test_clamp.py) inside each polarity's legal value range, and the fidelity sweep
drives values both below lo and above hi, so both clamp arms are exercised.

Output: ../vec/clamp.vec, one line per cycle:

    <rst> <i_input_uni> <o_uni> <i_input_bi> <o_bi>   (each 0/1, space-separated)

`rst`=1 marks the first cycle of each independent value's stream (the RTL
testbench pulses i_rst_n low there, returning to the exact post-reset() state).

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_clamp.py
"""
from pathlib import Path

import torch

from napl.sim.operation import clamp, encode
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "clamp.vec"
PARAMS = ROOT / "vec" / "clamp_params.vh"

# Encoder/band settings mirror test_clamp.py: a fixed band inside each polarity's
# legal value range, and the full legal sweep so values below lo and above hi are
# both driven through the counter.
TIMESTEP = 256
FRACWIDTH = 8
BOUNDS = {"bipolar": (-0.5, 0.5), "unipolar": (0.25, 0.75)}
SWEEP = {
    "bipolar": torch.linspace(-1.0, 1.0, 128),
    "unipolar": torch.linspace(0.0, 1.0, 128),
}


def grid_int(value):
    """Quantize a band bound to the integer grid the RTL parameters carry."""
    return round(value * (1 << FRACWIDTH))


def run(polarity):
    """Drive the model per value stream from reset; return per-cycle (input, output).

    Also report whether the run reached each clamp arm (a/t below lo and above hi),
    so the caller can require both arms to be exercised by the vectors.
    """
    lo, hi = BOUNDS[polarity]
    lo_i, hi_i = grid_int(lo), grid_int(hi)
    two_f = 1 << FRACWIDTH
    model = clamp({"polarity": polarity, "lo": lo, "hi": hi, "fracwidth": FRACWIDTH})
    segments = []
    below_hit = above_hit = 0
    for value in SWEEP[polarity]:
        enc = encode({"polarity": polarity, "timestep": TIMESTEP, "generator": "sobol", "dim": 1})
        enc.reset()
        model.reset()
        rows = []
        for _ in range(TIMESTEP):
            spike = enc(value)
            out = int(model(spike).item())
            rows.append((int(spike.item()), out))
            # Track which clamp arm the running estimate acc/t landed on.
            acc = int(model.accumulator.item())
            t = model.timestep_cur
            if acc * two_f < lo_i * t:
                below_hit += 1
            elif acc * two_f > hi_i * t:
                above_hit += 1
        segments.append(rows)
    return segments, model.hw.pp_delay, below_hit, above_hit


def main():
    seg_uni, pp_uni, below_uni, above_uni = run("unipolar")
    seg_bi, pp_bi, below_bi, above_bi = run("bipolar")
    assert pp_uni == pp_bi, f"polarity pp_delay disagree: {pp_uni} vs {pp_bi}"
    assert len(seg_uni) == len(seg_bi)
    # Both clamp arms must be exercised, per polarity, or the vectors would leave
    # one saturating branch of the RTL unchecked.
    for name, below, above in (("unipolar", below_uni, above_uni), ("bipolar", below_bi, above_bi)):
        assert below > 0 and above > 0, f"{name} vectors miss a clamp arm: below={below} above={above}"

    # Resolve each elaborated variant's mapping entry and require it to reproduce
    # the params these vectors were built with, so make test gates the mapping too.
    for polarity in ("unipolar", "bipolar"):
        lo, hi = BOUNDS[polarity]
        binding = translate_node({"class": "clamp", "config": {
            "polarity": polarity, "lo": lo, "hi": hi,
            "fracwidth": FRACWIDTH, "timestep": TIMESTEP}})
        expected = {"FRACWIDTH": FRACWIDTH, "LO": grid_int(lo),
                    "HI": grid_int(hi), "TIMESTEP": TIMESTEP}
        assert binding.parameters == expected, \
            f"mapping clamp_{polarity} resolves {binding.parameters}, not {expected}"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_LO_UNI {grid_int(BOUNDS['unipolar'][0])}\n"
        f"`define GEN_HI_UNI {grid_int(BOUNDS['unipolar'][1])}\n"
        f"`define GEN_LO_BI {grid_int(BOUNDS['bipolar'][0])}\n"
        f"`define GEN_HI_BI {grid_int(BOUNDS['bipolar'][1])}\n"
        f"`define GEN_FRACWIDTH {FRACWIDTH}\n"
        f"`define GEN_TIMESTEP {TIMESTEP}\n"
        f"`define GEN_PP_DELAY {pp_uni}\n"
    )

    rows = 0
    with VEC.open("w") as output:
        for segment_uni, segment_bi in zip(seg_uni, seg_bi):
            for cycle, ((in_uni, out_uni), (in_bi, out_bi)) in enumerate(zip(segment_uni, segment_bi)):
                reset = int(cycle == 0)
                output.write(f"{reset} {in_uni} {out_uni} {in_bi} {out_bi}\n")
                rows += 1

    print(
        f"wrote {VEC} ({rows} vectors, {len(seg_uni)} reset segments) and {PARAMS} "
        f"(LO_UNI={grid_int(BOUNDS['unipolar'][0])} HI_UNI={grid_int(BOUNDS['unipolar'][1])} "
        f"LO_BI={grid_int(BOUNDS['bipolar'][0])} HI_BI={grid_int(BOUNDS['bipolar'][1])} "
        f"FRACWIDTH={FRACWIDTH} TIMESTEP={TIMESTEP} PP_DELAY={pp_uni})"
    )


if __name__ == "__main__":
    main()
