"""
Generate golden test vectors for the clamp_comp RTL straight from napl's functional
Python model (napl.sim.operation.clamp_comp) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

clamp_comp chains two stream comparators: a maximum selector places the floor and a
minimum selector places the ceiling on its result. The band is a construction
constant the operation encodes itself, so the RTL instantiates one encode cell per
bound alongside the max and min cells. One bare module serves both polarities: the
selectors compare spike counts and never read the polarity, which reaches the
circuit only as the two bound probability codes, so the two polarity DUTs the
testbench elaborates are the same module carrying different LO_CODE and HI_CODE.

The stimulus mirrors tests/operation/test_clamp_comp.py: every swept value is a
point of a GRID_LEN grid, with bipolar entries on even multiples of one threshold
step and unipolar entries on odd ones, so no unipolar stream repeats a bipolar one.
check_streams_distinct() requires that separation on the emitted streams rather than
on the requested rates.

Each segment runs the full 2**SEQ_W period of the bound sequences from reset, so no
bound index goes undriven; a segment windowed shorter would leave the tail of that
period untested. An inverted band is not covered because the class rejects one:
check_inverted_rejected() runs that rejection.

Output: ../vec/clamp_comp.vec, one line per cycle:

    <rst> <i_input_uni> <o_uni> <i_input_bi> <o_bi>   (each 0/1, space-separated)

`rst`=1 marks the first cycle of each segment (the RTL testbench pulses i_rst_n low
there, returning to the exact post-reset() state).
../vec/clamp_comp_{lo,hi}.hex hold the Sobol direction vectors of the two bound
streams, recovered from the sequences the model itself draws.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_clamp_comp.py
"""
from pathlib import Path

import torch

from napl.sim.operation import clamp_comp, encode, gen_num_seq
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "clamp_comp.vec"
PARAMS = ROOT / "vec" / "clamp_comp_params.vh"

# Stimulus settings mirror tests/operation/test_clamp_comp.py.
GRID_LEN = 256
INPUT_DIM = 1
BOUND_DIM = 2
BOUNDS = {"bipolar": (-0.5, 0.5), "unipolar": (0.125, 0.6875)}
SWEEP_LEN = 33


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


def bound_code(polarity, bound, seq_width):
    """Return a bound's probability code in units of 2**-seq_width."""
    probability = bound if polarity == "unipolar" else (bound + 1.0) / 2.0
    code = round(probability * 2 ** seq_width)
    assert probability * 2 ** seq_width == code, \
        f"{polarity} bound {bound} is off the 2**-{seq_width} threshold grid"
    return code


def segments(polarity, timestep):
    """Return (stream, output) per swept value, and the model that produced them."""
    lo, hi = BOUNDS[polarity]
    model = clamp_comp({"polarity": polarity, "lo": lo, "hi": hi, "dim": BOUND_DIM})
    rows = []
    for step in sweep_steps(polarity):
        encoder = encode({"polarity": polarity, "timestep": timestep,
                          "generator": "sobol", "dim": INPUT_DIM})
        encoder.reset()
        model.reset()
        stream = []
        output = []
        for _ in range(timestep):
            spike = encoder(torch.tensor(grid_value(polarity, step)))
            stream.append(int(spike.item()))
            output.append(int(model(spike).item()))
        rows.append((stream, output))
    return rows, model


def check_band(rows, model):
    """Require the two bipolar rails to place the band the bound streams carry.

    The bipolar sweep runs rail to rail, so its first segment is the all-zero stream
    and its last the all-one stream. On the first the floor selector holds on the lo
    stream every timestep and on the last the ceiling selector holds on the hi
    stream, so each output is that bound stream itself, cycle by cycle from reset.
    A selector that stopped placing its bound moves the matching stream.
    """
    for name, (stream, output), bits, spike in (
        ("floor", rows[0], model.lo_bits, 0), ("ceiling", rows[-1], model.hi_bits, 1)
    ):
        assert all(item == spike for item in stream), \
            f"the {name} segment is not the all-{spike} stream"
        target = [int(bit) for bit in bits.tolist()]
        assert output == target, \
            f"the {name} selector emitted {sum(output)} spikes against a bound stream " \
            f"carrying {sum(target)}, departing from it at cycle " \
            f"{next(i for i, (a, b) in enumerate(zip(output, target)) if a != b)}"


def check_inverted_rejected():
    """Require the class to reject an inverted band, which is why none is swept."""
    for polarity, (lo, hi) in BOUNDS.items():
        try:
            clamp_comp({"polarity": polarity, "lo": hi, "hi": lo, "dim": BOUND_DIM})
        except AssertionError:
            continue
        raise AssertionError(f"clamp_comp accepted the inverted {polarity} band")


def check_streams_distinct(blocks):
    """Require every segment to carry an input stream no other segment carries.

    A duplicate segment is a vector row that proves nothing the file already proved,
    and a rate is all a Sobol encoder reads, so two polarities asking for the same
    rates emit the same spikes. Comparing the emitted streams catches that whatever
    the requested values were.
    """
    seen = {}
    for name, rows in blocks:
        for index, (stream, _) in enumerate(rows):
            key = tuple(stream)
            assert key not in seen, \
                f"{name} segment {index} repeats the stimulus of {seen[key]}"
            seen[key] = f"{name} segment {index}"
    return len(seen)


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


def write_dirvec(seq_width):
    """Emit the direction-vector tables the RTL encode cells generate from.

    The two bound streams sit on consecutive Sobol dimensions, and the tables are
    asserted to differ. The sequences do not depend on the band or the polarity, so
    one file per dimension serves both polarity DUTs.
    """
    tables = {}
    for name, offset in (("lo", 0), ("hi", 1)):
        num_seq = gen_num_seq({"width": seq_width, "generator": "sobol",
                               "dim": BOUND_DIM + offset})
        rows = dirvec_rows(num_seq, seq_width, f"{name} bound sequence")
        for other, other_rows in tables.items():
            assert rows != other_rows, f"the {name} bound sequence shares {other}"
        tables[name] = rows
        (ROOT / "vec" / f"clamp_comp_{name}.hex").write_text("\n".join(rows) + "\n")


def check_mapping(seq_width):
    """Require each polarity's mapping entry to resolve the params these vectors use."""
    for polarity, (lo, hi) in BOUNDS.items():
        binding = translate_node({"class": "clamp_comp", "config": {
            "polarity": polarity, "lo": lo, "hi": hi, "dim": BOUND_DIM}})
        assert binding.rtl_module == "clamp_comp", \
            f"mapping resolves clamp_comp to {binding.rtl_module}"
        expected = {"SEQ_W": seq_width,
                    "LO_CODE": bound_code(polarity, lo, seq_width),
                    "HI_CODE": bound_code(polarity, hi, seq_width)}
        assert binding.parameters == expected, \
            f"mapping clamp_comp resolves {binding.parameters} for {polarity}, not {expected}"


def main():
    # The segment length is the bound period the class fixes, so one segment drives
    # every bound index; a shorter segment would leave that period's tail unreached.
    lo, hi = BOUNDS["bipolar"]
    period = clamp_comp({"polarity": "bipolar", "lo": lo, "hi": hi,
                         "dim": BOUND_DIM}).constant_len
    seq_width = period.bit_length() - 1
    assert 2 ** seq_width == period, f"the bound period {period} is not a power of two"

    rows_bi, model_bi = segments("bipolar", period)
    rows_uni, model_uni = segments("unipolar", period)
    assert model_uni.constant_len == period, \
        f"the unipolar bound period is {model_uni.constant_len}, not {period}"
    assert model_uni.hw.pp_delay == model_bi.hw.pp_delay, \
        f"polarity pp_delay disagree: {model_uni.hw.pp_delay} vs {model_bi.hw.pp_delay}"

    check_band(rows_bi, model_bi)
    check_inverted_rejected()
    distinct = check_streams_distinct(
        [("unipolar sweep", rows_uni), ("bipolar sweep", rows_bi)])

    VEC.parent.mkdir(parents=True, exist_ok=True)
    write_dirvec(seq_width)
    check_mapping(seq_width)

    count = 0
    with VEC.open("w") as output:
        for (stream_uni, out_uni), (stream_bi, out_bi) in zip(rows_uni, rows_bi):
            for cycle, (in_uni, o_uni, in_bi, o_bi) in enumerate(
                    zip(stream_uni, out_uni, stream_bi, out_bi)):
                output.write(f"{int(cycle == 0)} {in_uni} {o_uni} {in_bi} {o_bi}\n")
                count += 1

    codes = {f"{name}_{key}": bound_code(polarity, bound, seq_width)
             for polarity, key in (("unipolar", "UNI"), ("bipolar", "BI"))
             for name, bound in zip(("LO_CODE", "HI_CODE"), BOUNDS[polarity])}
    PARAMS.write_text(
        f"`define GEN_SEQ_W {seq_width}\n"
        + "".join(f"`define GEN_{name} {value}\n" for name, value in sorted(codes.items()))
        + f"`define GEN_PP_DELAY {model_bi.hw.pp_delay}\n"
          f"`define GEN_VECTORS {count}\n"
    )
    print(f"wrote {VEC} ({count} vectors over {distinct} distinct segments) and {PARAMS} "
          f"(SEQ_W={seq_width} {codes} PP_DELAY={model_bi.hw.pp_delay})")


if __name__ == "__main__":
    main()
