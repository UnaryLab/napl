"""
Generate golden test vectors for the encode RTL straight from napl's functional
Python model (napl.sim.operation.encode), so the testbench checks the Verilog
against the actual simulator rather than a hand-derived truth table.

encode compares the encoded probability p against a periodic number sequence:
p = x for unipolar, p = (x+1)/2 for bipolar, and s_t = 1{p > q[(t-1) % L]}. The
polarity only selects how p is derived from the input value, so a single bare
`encode` module taking p covers both; this generator emits blocks for both
polarities through that one module.

The number sequence is produced online in hardware by a Sobol generator running
the Antonov-Saleev gray-code recurrence x_{n+1} = x_n ^ v[l(n)], where l(n) is
the position of the least significant zero of n. This generator recovers the
direction-vector table v of each covered Sobol dimension from the model's own
num_seq, asserts the recurrence reproduces the whole 2**WIDTH-point sequence and
returns to 0 on the wrap, and writes it to ../vec/encode_dirvec_d<dim>.hex.

Fixed point: sequence values are k/2**WIDTH, and bipolar p lands on a half-step of
that grid, so p is carried in FRAC = WIDTH+1 fractional bits, where every driven
value is exactly representable and the integer compare reproduces the model's
float compare bit for bit.

The remaining hardware number-sequence generators are checked the same way. lfsr
and lfsr_ext are Fibonacci LFSRs whose tap mask is recovered from the model's own
num_seq by search and asserted to reproduce it over a full period; temporalseq is
the ascending ramp num_seq[i] = i/2**WIDTH. Their golden values sit in a second
file beside a WIDTH=1 encode instance, which is what walks the two-entry table
branch of sobol.v that no WIDTH=8 instance reaches.

Output: ../vec/encode.vec, one line per timestep:

    <rst> <dim> <i_input> <o_spike>   (rst=1 means "reset BEFORE this cycle",
                                       dim selects the Sobol dimension under
                                       test, i_input is the decimal p code, and
                                       o_spike is 0/1)

and ../vec/encode_gen.vec, one line per timestep:

    <rst> <lfsr> <lfsr_ext> <temporalseq> <w1_input> <w1_spike>
          <mode_input> <lfsr_spike> <lfsr_ext_spike> <tc_spike>

with the three sequence values as decimal codes scaled by 2**WIDTH, then the
WIDTH=1 encode instance's p code and spike, then one p code driven into three
full encode instances elaborated at GENERATOR 1, 2, and 3 and their spikes. The
sequence columns pin each generator circuit; the spike columns pin encode.v's
GENERATOR selection, which is what a configuration translates into.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_encode.py
"""
from pathlib import Path

import torch
from napl.sim.operation import encode
from napl.syn import translate_node


VEC = Path(__file__).resolve().parent.parent / "vec" / "encode.vec"
GEN_VEC = Path(__file__).resolve().parent.parent / "vec" / "encode_gen.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "encode_params.vh"
DIRVEC_DIR = Path(__file__).resolve().parent.parent / "vec"

# Sizing and generator settings mirror test_encode.py's main case.
TIMESTEP = 256
GENERATOR = "sobol"

# The narrowest stream the mapping accepts: WIDTH 1, which is the two-entry table
# branch of sobol.v. No other elaborated instance reaches it.
TIMESTEP_W1 = 2

# Periods of the sequence-generator block: two from reset, then one more after a
# reset taken from the dirtied state the first two leave behind.
GEN_PERIODS = 3

# Width range mapping.yaml's tap table covers, walked against the model below.
TAPS_WIDTH_LO = 2
TAPS_WIDTH_HI = 12

# gen_num_seq generators reaching a sequence circuit other than Sobol's, each
# driven through a full encode instance so the GENERATOR selection is what the
# co-simulation checks. `sys` has no circuit and stays rejected by the mapping.
NON_SOBOL_MODES = ("lfsr", "lfsr_ext", "tc")

# Unipolar p codes cycled through the non-Sobol encoders, on the 1/2**FRAC grid.
MODE_CODES = (0, 1, 128, 256, 384, 511, 512)

# Sobol dimensions the testbench elaborates, one direction-vector file each. The
# testbench builds the file names from `GEN_DIMS, so the set stays 1..len(DIMS).
DIMS = (1, 2, 3, 4)

# Values whose probability sits exactly on a sequence value: a `>=` comparator
# would emit one extra spike on these, so they must be driven.
TIE_VALUES = {"bipolar": [0.0, 0.5], "unipolar": [0.5, 0.25]}


def cfg(polarity, dim):
    return {"polarity": polarity, "timestep": TIMESTEP, "generator": GENERATOR, "dim": dim}


def values(polarity, frac):
    """Driven input values, quantized onto the 1/2**frac probability grid."""
    lo = -1.0 if polarity == "bipolar" else 0.0
    raw = TIE_VALUES[polarity] + [lo, 1.0, (lo + 1.0) / 4]
    g = torch.Generator().manual_seed(0)
    raw += (lo + (1.0 - lo) * torch.rand(3, generator=g)).tolist()

    out, seen = [], set()
    for x in raw:
        p = (x + 1.0) / 2 if polarity == "bipolar" else x
        code = round(p * 2 ** frac)
        if code in seen:
            continue
        seen.add(code)
        p_q = code / 2 ** frac
        out.append((2 * p_q - 1 if polarity == "bipolar" else p_q, code))
    return out


def lsz(n, width):
    """Position of the least significant zero of the width-bit counter value n."""
    if n == 2 ** width - 1:
        # The all-ones state carries v[width-1], which returns the sequence to 0.
        return width - 1
    return ((~n & (n + 1)).bit_length() - 1)


def write_dirvec(model, dim, path=None):
    """Emit the dimension's direction vectors, recovered from the model sequence."""
    width = model.width
    period = 2 ** width
    seq = model.num_seq.detach().float().reshape(-1)
    assert seq.numel() == model.len == period, f"num_seq length {seq.numel()} != {period}"

    codes = []
    for i in range(period):
        scaled = seq[i].item() * period
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f"num_seq[{i}]={seq[i].item()} off the 1/2**{width} grid"
        codes.append(code)
    assert codes[0] == 0, f"dim {dim} sequence starts at {codes[0]}, not the post-reset 0"

    vecs = [codes[2 ** k] ^ codes[2 ** k - 1] for k in range(width)]

    x = 0
    for n in range(period):
        assert x == codes[n], (
            f"dim {dim} unsupported: the gray-code recurrence gives {x} at index {n}, "
            f"model num_seq gives {codes[n]}"
        )
        x ^= vecs[lsz(n, width)]
    assert x == 0, f"dim {dim} recurrence returns to {x} on the wrap, not 0"

    path = path or DIRVEC_DIR / f"encode_dirvec_d{dim}.hex"
    path.write_text("\n".join(f"{v:0{width}b}" for v in vecs) + "\n")
    return path


def write_sweep_dirvecs():
    """Emit a dimension-1 table per swept width, sized WIDTH words of WIDTH bits.

    The testbench elaborates one sobol instance per width in 2..TAPS_WIDTH_HI, and
    each reads its table at its own width, so one table per width is what makes
    every $readmemb range match its file.
    """
    paths = []
    for width in range(2, TAPS_WIDTH_HI + 1):
        model = encode({"polarity": "bipolar", "timestep": 2 ** width,
                        "generator": GENERATOR, "dim": 1})
        assert model.width == width, f"timestep 2**{width} gave width {model.width}"
        paths.append(write_dirvec(model, 1, DIRVEC_DIR / f"encode_dv_w{width:02d}.hex"))
    return paths


def seq_codes(model):
    """The model's number sequence as integer codes scaled by 2**width."""
    period = 2 ** model.width
    seq = model.num_seq.detach().float().reshape(-1)
    assert seq.numel() == period, f"num_seq length {seq.numel()} != {period}"
    codes = []
    for i in range(period):
        scaled = seq[i].item() * period
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, \
            f"num_seq[{i}]={seq[i].item()} off the 1/2**{model.width} grid"
        codes.append(code)
    return codes


def parity(value):
    """XOR of the set bits of value."""
    out = 0
    while value:
        out ^= value & 1
        value >>= 1
    return out


def replays(codes, width, mask, extended):
    """Whether the RTL recurrence at this tap mask walks the model codes exactly.

    `extended` selects the de Bruijn feedback of lfsr_ext.v, which inverts the tap
    XOR on the two states whose upper bits are zero and so inserts the all-zero
    state; the plain form instead holds on the last index, where the model repeats
    its opening state.
    """
    period = 2 ** width
    if not extended and codes[period - 1] != codes[0]:
        return False

    state = codes[0]
    for i in range(period):
        if state != codes[i]:
            return False
        if extended or i < period - 1:
            feedback = parity(state & mask)
            if extended and (state >> 1) == 0:
                feedback ^= 1
            state = (feedback << (width - 1)) | (state >> 1)
    return (not extended) or state == codes[0]


def recover_taps(codes, width, extended):
    """Recover the tap mask whose LFSR walk reproduces the model codes.

    Searching the mask and replaying the whole period against the model sequence
    makes the emitted parameter true by construction: a mask that did not
    reproduce the model would leave the search empty.
    """
    found = [mask for mask in range(2 ** width) if replays(codes, width, mask, extended)]
    assert found, (
        f"no tap mask reproduces the {'extended ' if extended else ''}lfsr sequence "
        f"at width {width}; the RTL recurrence does not cover this model sequence"
    )
    return found[0]


def check_tap_table():
    """Replay mapping.yaml's whole tap table against the model, width by width.

    Only the elaborated width reaches a vector row, so without this walk a wrong
    row for any other width would translate into wrong hardware unnoticed.
    """
    for width in range(TAPS_WIDTH_LO, TAPS_WIDTH_HI + 1):
        for generator, extended in (("lfsr", False), ("lfsr_ext", True)):
            config = {"polarity": "unipolar", "timestep": 2 ** width, "generator": generator}
            resolved = translate_node({"class": "encode", "config": config}).parameters
            codes = seq_codes(encode(config))
            assert resolved["WIDTH"] == width, \
                f"mapping resolves WIDTH {resolved['WIDTH']} at timestep {2 ** width}"
            assert resolved["SEED"] == codes[0], \
                f"mapping resolves SEED {resolved['SEED']}, model opens on {codes[0]}"
            assert replays(codes, width, resolved["TAPS"], extended), (
                f"mapping resolves TAPS {resolved['TAPS']} for {generator} at width {width}, "
                f"which does not walk the model sequence"
            )


def w1_pairs(frac):
    """Every p code the WIDTH=1 encode instance can be driven with, as (value, code)."""
    return [(2 * code / 2 ** frac - 1, code) for code in range(2 ** frac + 1)]


def check_parameters(config, expected):
    """Require the mapping entry to resolve the parameters these vectors were built with."""
    resolved = translate_node({"class": "encode", "config": config}).parameters
    assert resolved == expected, \
        f"mapping encode {config} resolves {resolved}, not {expected}"
    return resolved


def mode_config(generator, timestep=TIMESTEP):
    """One unipolar encode configuration for a non-Sobol sequence generator."""
    return {"polarity": "unipolar", "timestep": timestep, "generator": generator}


def write_gen_vectors(width, frac, taps, seed):
    """Emit the sequence-generator rows beside the encode instances they drive.

    Each non-Sobol generator is compared twice per row: as a bare o_rand value,
    which pins the sequence itself, and through a full encode instance, which is
    what pins the GENERATOR selection inside encode.v.
    """
    period = 2 ** width
    codes = {name: seq_codes(encode(mode_config(name))) for name in NON_SOBOL_MODES}
    assert codes["lfsr"][0] == seed and codes["lfsr_ext"][0] == seed, \
        "the emitted reset state must be the model sequences' opening value"
    assert codes["tc"] == list(range(period)), \
        f"tc num_seq is not the ascending ramp: {codes['tc'][:4]}..."
    for name, extended in (("lfsr", False), ("lfsr_ext", True)):
        assert replays(codes[name], width, taps, extended), \
            f"the emitted TAPS {taps} does not walk the {name} model sequence"

    w1 = encode({"polarity": "bipolar", "timestep": TIMESTEP_W1,
                 "generator": GENERATOR, "dim": 1})
    frac_w1 = w1.width + 1
    pairs = w1_pairs(frac_w1)
    check_parameters({"polarity": "bipolar", "timestep": TIMESTEP_W1,
                      "generator": GENERATOR, "dim": 1},
                     {"WIDTH": w1.width, "FRAC": frac_w1, "GENERATOR": 0, "TAPS": 0,
                      "SEED": 1, "DIRVEC_FILE": "vec/encode_dirvec_d1.hex"})

    models = {name: encode(mode_config(name)) for name in NON_SOBOL_MODES}
    rows = []
    for block in range(GEN_PERIODS):
        # Block 1 runs straight on from block 0; block 2 resets from the state the
        # first two dirtied and must replay block 0's values.
        if block != 1:
            w1.reset()
            for model in models.values():
                model.reset()
        for t in range(period):
            rst = 1 if (t == 0 and block != 1) else 0
            value, code = pairs[t % len(pairs)]
            spike = int(w1(torch.tensor(float(value)).type(w1.num_seq.dtype)).item())
            mode_code = MODE_CODES[t % len(MODE_CODES)]
            probability = torch.tensor(mode_code / 2 ** frac).type(w1.num_seq.dtype)
            spikes = " ".join(str(int(models[name](probability).item()))
                              for name in NON_SOBOL_MODES)
            sequence = " ".join(str(codes[name][t]) for name in NON_SOBOL_MODES)
            rows.append(f"{rst} {sequence} {code} {spike} {mode_code} {spikes}")

    assert len(rows) == GEN_PERIODS * period, \
        f"expected {GEN_PERIODS * period} generator rows, built {len(rows)}"
    GEN_VEC.write_text("\n".join(rows) + "\n")
    return rows, w1.width, frac_w1


def drive(model, dim, value, code, rows, first_rst):
    """Run the model over one full period and append its rows."""
    x = torch.tensor(float(value)).type(model.num_seq.dtype)
    for t in range(model.len):
        rst = 1 if (t == 0 and first_rst) else 0
        spike = int(model(x).item())
        rows.append(f"{rst} {dim} {code} {spike}")


def main():
    assert DIMS == tuple(range(1, len(DIMS) + 1)), \
        f"the testbench names its tables d1..d`GEN_DIMS, so DIMS must be contiguous from 1, got {DIMS}"
    assert len(DIMS) <= 9, \
        f"the testbench builds each table path with the single ASCII digit 8'd48 + d, " \
        f"so it cannot name more than 9 dimensions, got {len(DIMS)}"
    VEC.parent.mkdir(parents=True, exist_ok=True)

    ref = encode(cfg("bipolar", 1))
    frac = ref.width + 1
    dirvec_paths = [write_dirvec(encode(cfg("bipolar", dim)), dim) for dim in DIMS]
    sweep_paths = write_sweep_dirvecs()

    # Resolve each elaborated dimension's mapping entry and require it to
    # reproduce the params these vectors were built with, so make test gates the
    # mapping too: the direction-vector table is what a wrong dim silently swaps.
    for dim, path in zip(DIMS, dirvec_paths):
        check_parameters(cfg("bipolar", dim),
                         {"WIDTH": ref.width, "FRAC": frac, "GENERATOR": 0, "TAPS": 0,
                          "SEED": 1, "DIRVEC_FILE": f"vec/{path.name}"})

    rows = []
    for dim in DIMS:
        for polarity in ("bipolar", "unipolar"):
            model = encode(cfg(polarity, dim))
            for value, code in values(polarity, frac):
                model.reset()
                drive(model, dim, value, code, rows, first_rst=True)

    # Reset from a dirtied counter must replay a fresh-reset run of the same input.
    for dim in DIMS:
        model = encode(cfg("bipolar", dim))
        model.reset()
        value, code = values("bipolar", frac)[0]
        x = torch.tensor(float(value)).type(model.num_seq.dtype)
        for _ in range(model.len // 3):
            model(x)
        model.reset()
        drive(model, dim, value, code, rows, first_rst=True)

    expected_rows = len(DIMS) * (len(values("bipolar", frac)) + len(values("unipolar", frac))
                                 + 1) * ref.len
    assert len(rows) == expected_rows, \
        f"expected {expected_rows} encode rows, built {len(rows)}"
    VEC.write_text("\n".join(rows) + "\n")

    # Every row of the mapping's tap table, not just the elaborated width.
    check_tap_table()

    # Each non-Sobol mode's own resolved parameters, which is what selects the
    # sequence circuit inside encode.v. The tap mask is recovered from the model
    # sequence by search and required to be the one the mapping resolves.
    codes = {name: seq_codes(encode(mode_config(name))) for name in NON_SOBOL_MODES}
    taps = recover_taps(codes["lfsr"], ref.width, extended=False)
    assert taps == recover_taps(codes["lfsr_ext"], ref.width, extended=True), \
        "the plain and extended cycles must share the model's feedback polynomial"
    modes = {}
    for index, name in enumerate(NON_SOBOL_MODES, start=1):
        expected_taps = taps if name != "tc" else 0
        modes[name] = check_parameters(
            mode_config(name),
            {"WIDTH": ref.width, "FRAC": frac, "GENERATOR": index, "TAPS": expected_taps,
             "SEED": 1, "DIRVEC_FILE": "vec/encode_dirvec_d1.hex"})

    gen_rows, width_w1, frac_w1 = write_gen_vectors(ref.width, frac, taps, codes["lfsr"][0])

    PARAMS.write_text(
        f"`define GEN_WIDTH {ref.width}\n"
        f"`define GEN_FRAC {frac}\n"
        f"`define GEN_PP_DELAY {ref.hw.pp_delay}\n"
        f"`define GEN_DIMS {len(DIMS)}\n"
        f"`define GEN_VECTORS {len(rows)}\n"
        f"`define GEN_TAPS {modes['lfsr']['TAPS']}\n"
        f"`define GEN_SEED {modes['lfsr']['SEED']}\n"
        f"`define GEN_MODE_LFSR {modes['lfsr']['GENERATOR']}\n"
        f"`define GEN_MODE_LFSR_EXT {modes['lfsr_ext']['GENERATOR']}\n"
        f"`define GEN_MODE_TC {modes['tc']['GENERATOR']}\n"
        f"`define GEN_WIDTH_W1 {width_w1}\n"
        f"`define GEN_FRAC_W1 {frac_w1}\n"
        f"`define GEN_SEQ_VECTORS {len(gen_rows)}\n"
        f"`define GEN_SWEEP_HI {TAPS_WIDTH_HI}\n"
    )
    print(f"wrote {VEC} ({len(rows)} vectors), {GEN_VEC} ({len(gen_rows)} vectors), {PARAMS} "
          f"(GEN_WIDTH={ref.width}, GEN_FRAC={frac}, GEN_DIMS={len(DIMS)}, "
          f"GEN_TAPS={taps}, GEN_WIDTH_W1={width_w1}, "
          f"modes {[(name, modes[name]['GENERATOR']) for name in NON_SOBOL_MODES]}), and "
          f"{len(dirvec_paths)} direction-vector files {[p.name for p in dirvec_paths]} "
          f"plus {len(sweep_paths)} width-sweep tables w02..w{TAPS_WIDTH_HI:02d}")


if __name__ == "__main__":
    main()
