"""
Generate golden test vectors for the clamp_sat RTL straight from napl's functional
Python model (napl.sim.operation.clamp_sat) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

clamp_sat chains three unit-scale bipolar add_scale stages whose saturations place
the band [lo, hi]. Each stage carries the stream, one Sobol constant stream, and
one fixed rail, so the RTL instantiates three encode cells and three
add_scale_bipolar cells. The unipolar variant runs the same bipolar circuit on
the band mapped to (2*lo - 1, 2*hi - 1), so there are two variants:
clamp_sat_unipolar and clamp_sat_bipolar. The sizing keys are the band [lo, hi] in grid
units and its fracwidth; the band is chosen (mirroring test_clamp_sat.py) inside each
polarity's legal value range, and the fidelity sweep drives values both below lo
and above hi, so both saturating stages are exercised.

Output: ../vec/clamp_sat.vec, one line per cycle:

    <rst> <i_input_uni> <o_uni> <i_input_bi> <o_bi>   (each 0/1, space-separated)

`rst`=1 marks the first cycle of each independent value's stream (the RTL
testbench pulses i_rst_n low there, returning to the exact post-reset() state).
../vec/clamp_sat_c{0,1,2}.hex hold the Sobol direction vectors of the three constant
streams, recovered from the sequences the model itself draws.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_clamp_sat.py
"""
from pathlib import Path

import torch

from napl.sim.operation import clamp_sat, encode, gen_num_seq
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "clamp_sat.vec"
PARAMS = ROOT / "vec" / "clamp_sat_params.vh"

# Band settings mirror test_clamp_sat.py: a fixed band inside each polarity's legal
# value range, and the full legal sweep so values below lo and above hi are both
# driven through the stages. Every segment restarts the constant streams from
# index 0, so a segment shorter than their period 2**(FRACWIDTH+1) leaves the
# tail of that period unreached by any row and hides the constant codes riding
# it; TIMESTEP is that period, so one segment covers every index.
FRACWIDTH = 8
TIMESTEP = 2 ** (FRACWIDTH + 1)
# The constant streams sit on dims DIM, DIM+1, DIM+2, clear of the input encoder's
# dimension 1.
DIM = 2
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

    The model is returned with the segments so the caller can pin the two
    band-placing saturations against the constants the stages carry.
    """
    lo, hi = BOUNDS[polarity]
    model = clamp_sat({"polarity": polarity, "lo": lo, "hi": hi,
                   "fracwidth": FRACWIDTH, "dim": DIM})
    segments = []
    for value in SWEEP[polarity]:
        enc = encode({"polarity": polarity, "timestep": TIMESTEP, "generator": "sobol", "dim": 1})
        enc.reset()
        model.reset()
        rows = []
        for _ in range(TIMESTEP):
            spike = enc(value)
            out = int(model(spike).item())
            rows.append((int(spike.item()), out))
        segments.append(rows)
    return segments, model.hw.pp_delay, model


def check_saturations(polarity, segments, model):
    """Require the vectors to exercise both saturations that place the band.

    The sweep runs from the bottom of the polarity's legal range to the top, so
    its first segment is the all-zero stream and its last the all-one stream. On
    the first the floor stage saturates every timestep and the output rate is lo;
    on the last the ceiling stage saturates every timestep and the shift stage
    passes its hi constant stream through unchanged. A saturation that is widened
    or removed changes both of these compared segments.
    """
    band_lo = model.lo if polarity == "bipolar" else 2.0 * model.lo - 1.0
    floor_segment = segments[0]
    assert all(spike == 0 for spike, _ in floor_segment), \
        f"{polarity} first segment is not the all-zero stream"
    floor_count = sum(out for _, out in floor_segment)
    expected = round(TIMESTEP * (band_lo + 1.0) / 2.0)
    assert floor_count == expected, \
        f"{polarity} floor saturation emitted {floor_count} spikes, not {expected}"

    ceiling_segment = segments[-1]
    assert all(spike == 1 for spike, _ in ceiling_segment), \
        f"{polarity} last segment is not the all-one stream"
    for step, (_, out) in enumerate(ceiling_segment):
        expected = int(model.constant_bits[2][step % model.constant_len])
        assert out == expected, \
            f"{polarity} ceiling saturation broke at timestep {step}: {out} not {expected}"


def dirvec_rows(num_seq, width, name):
    """Direction vectors of a Sobol sequence, checked against the model's num_seq.

    The hardware generator runs the Antonov-Saleev gray-code recurrence
    x_{n+1} = x_n ^ v[l(n)], where l(n) is the position of the least significant
    zero of the width-bit counter n. The table v is recovered from the model's own
    sequence and replayed here over the whole period, so a sequence the recurrence
    does not reproduce fails the generator instead of the co-simulation.
    """
    period = 2 ** width
    values = num_seq.detach().float().reshape(-1)
    assert values.numel() == period, f"{name} holds {values.numel()} points, not {period}"
    codes = []
    for index in range(period):
        scaled = values[index].item() * period
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f"{name}[{index}] is off the 1/{period} grid"
        codes.append(code)
    assert codes[0] == 0, f"{name} starts at {codes[0]}, not the post-reset 0"

    vectors = [codes[2 ** k] ^ codes[2 ** k - 1] for k in range(width)]
    state = 0
    for index in range(period):
        assert state == codes[index], \
            f"{name} is not a gray-code Sobol sequence: the recurrence gives {state} " \
            f"at index {index}, the model gives {codes[index]}"
        # The all-ones counter state takes the top position, which returns to 0.
        position = width - 1 if index == period - 1 else (~index & (index + 1)).bit_length() - 1
        state ^= vectors[position]
    assert state == 0, f"{name} returns to {state} on the wrap, not 0"
    return [f"{vector:0{width}b}" for vector in vectors]


def write_dirvec():
    """Emit the direction-vector tables the RTL encode cells generate from.

    The three constant streams sit on consecutive Sobol dimensions, and the tables
    are asserted to differ. The sequences do not depend on the band or the
    polarity, so one file per dimension serves both polarity DUTs.
    """
    written = {}
    for index in range(3):
        num_seq = gen_num_seq({"width": FRACWIDTH + 1, "generator": "sobol", "dim": DIM + index})
        rows = dirvec_rows(num_seq, FRACWIDTH + 1, f"constant sequence {index}")
        for other, other_rows in written.items():
            assert rows != other_rows, f"constant sequence {index} shares {other}"
        written[index] = rows
        (ROOT / "vec" / f"clamp_sat_c{index}.hex").write_text("\n".join(rows) + "\n")


def check_mapping():
    """Require each variant's mapping entry to resolve the params these vectors use."""
    for polarity in ("unipolar", "bipolar"):
        lo, hi = BOUNDS[polarity]
        binding = translate_node({"class": "clamp_sat", "config": {
            "polarity": polarity, "lo": lo, "hi": hi,
            "fracwidth": FRACWIDTH, "dim": DIM}})
        expected = {"FRACWIDTH": FRACWIDTH, "LO": grid_int(lo), "HI": grid_int(hi)}
        assert binding.parameters == expected, \
            f"mapping clamp_sat_{polarity} resolves {binding.parameters}, not {expected}"


def main():
    seg_uni, pp_uni, model_uni = run("unipolar")
    seg_bi, pp_bi, model_bi = run("bipolar")
    assert pp_uni == pp_bi, f"polarity pp_delay disagree: {pp_uni} vs {pp_bi}"
    assert len(seg_uni) == len(seg_bi)
    check_saturations("unipolar", seg_uni, model_uni)
    check_saturations("bipolar", seg_bi, model_bi)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    write_dirvec()
    check_mapping()

    rows = 0
    with VEC.open("w") as output:
        for segment_uni, segment_bi in zip(seg_uni, seg_bi):
            for cycle, ((in_uni, out_uni), (in_bi, out_bi)) in enumerate(zip(segment_uni, segment_bi)):
                reset = int(cycle == 0)
                output.write(f"{reset} {in_uni} {out_uni} {in_bi} {out_bi}\n")
                rows += 1

    PARAMS.write_text(
        f"`define GEN_LO_UNI {grid_int(BOUNDS['unipolar'][0])}\n"
        f"`define GEN_HI_UNI {grid_int(BOUNDS['unipolar'][1])}\n"
        f"`define GEN_LO_BI {grid_int(BOUNDS['bipolar'][0])}\n"
        f"`define GEN_HI_BI {grid_int(BOUNDS['bipolar'][1])}\n"
        f"`define GEN_FRACWIDTH {FRACWIDTH}\n"
        f"`define GEN_PP_DELAY {pp_uni}\n"
        f"`define GEN_VECTORS {rows}\n"
    )

    print(
        f"wrote {VEC} ({rows} vectors, {len(seg_uni)} reset segments) and {PARAMS} "
        f"(LO_UNI={grid_int(BOUNDS['unipolar'][0])} HI_UNI={grid_int(BOUNDS['unipolar'][1])} "
        f"LO_BI={grid_int(BOUNDS['bipolar'][0])} HI_BI={grid_int(BOUNDS['bipolar'][1])} "
        f"FRACWIDTH={FRACWIDTH} PP_DELAY={pp_uni})"
    )


if __name__ == "__main__":
    main()
