"""
Generate golden test vectors for the encode RTL straight from napl's functional
Python model (napl.sim.operation.encode), so the testbench checks the Verilog
against the actual simulator rather than a hand-derived truth table.

encode compares the encoded probability p against a periodic number sequence:
p = x for unipolar, p = (x+1)/2 for bipolar, and s_t = 1{p > q[(t-1) % L]}. The
polarity only selects how p is derived from the input value, so a single bare
`encode` module taking p covers both; this generator emits blocks for both
polarities through that one module.

The number sequence itself is data, not logic: it is emitted here from the model
into ../vec/encode_rom.hex and loaded by the RTL with $readmemb, which covers the
sobol / lfsr / sys / temporal generators uniformly.

Fixed point: sequence values are k/2**WIDTH, and bipolar p lands on a half-step of
that grid, so both are carried in FRAC = WIDTH+1 fractional bits, where every
driven value is exactly representable and the integer compare reproduces the
model's float compare bit for bit.

Output: ../vec/encode.vec, one line per timestep:

    <rst> <i_input> <o_spike>   (rst=1 means "reset BEFORE this cycle",
                                 i_input is the decimal p code, o_spike is 0/1)

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_encode.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import encode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import require_seeded_sys

VEC = Path(__file__).resolve().parent.parent / "vec" / "encode.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "encode_params.vh"
ROM = Path(__file__).resolve().parent.parent / "vec" / "encode_rom.hex"

# Sizing and generator settings mirror test_encode.py's main case.
TIMESTEP = 256
GENERATOR = "sobol"

# Values whose probability sits exactly on a sequence value: a `>=` comparator
# would emit one extra spike on these, so they must be driven.
TIE_VALUES = {"bipolar": [0.0, 0.5], "unipolar": [0.5, 0.25]}


def cfg(polarity):
    return {"polarity": polarity, "timestep": TIMESTEP, "generator": GENERATOR, "dim": 1}


require_seeded_sys(cfg("unipolar"), cfg("bipolar"))


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


def write_rom(model, frac):
    """Emit the model's number sequence as frac-bit binary ROM lines."""
    seq = model.num_seq.detach().float().reshape(-1)
    assert seq.numel() == model.len, f"num_seq length {seq.numel()} != {model.len}"
    lines = []
    for i in range(model.len):
        scaled = seq[i].item() * 2 ** frac
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f"num_seq[{i}]={seq[i].item()} off the 1/2**{frac} grid"
        assert 0 <= code < 2 ** frac, f"sequence code {code} out of [0, 2**{frac})"
        lines.append(f"{code:0{frac}b}")
    ROM.write_text("\n".join(lines) + "\n")
    return len(lines)


def drive(model, value, code, rows, first_rst):
    """Run the model over one full period and append its rows."""
    x = torch.tensor(float(value)).type(model.num_seq.dtype)
    for t in range(model.len):
        rst = 1 if (t == 0 and first_rst) else 0
        spike = int(model(x).item())
        rows.append(f"{rst} {code} {spike}")


def main():
    VEC.parent.mkdir(parents=True, exist_ok=True)

    ref = encode(cfg("bipolar"))
    frac = ref.width + 1
    rom_lines = write_rom(ref, frac)
    PARAMS.write_text(
        f"`define GEN_WIDTH {ref.width}\n"
        f"`define GEN_FRAC {frac}\n"
        f"`define GEN_PP_DELAY {ref.hw.pp_delay}\n"
    )

    rows = []
    for polarity in ("bipolar", "unipolar"):
        model = encode(cfg(polarity))
        for value, code in values(polarity, frac):
            model.reset()
            drive(model, value, code, rows, first_rst=True)

    # Reset from a dirtied counter must replay a fresh-reset run of the same input.
    model = encode(cfg("bipolar"))
    model.reset()
    value, code = values("bipolar", frac)[0]
    x = torch.tensor(float(value)).type(model.num_seq.dtype)
    for _ in range(model.len // 3):
        model(x)
    model.reset()
    drive(model, value, code, rows, first_rst=True)

    VEC.write_text("\n".join(rows) + "\n")
    print(f"wrote {VEC} ({len(rows)} vectors), {PARAMS} "
          f"(GEN_WIDTH={ref.width}, GEN_FRAC={frac}), and {ROM} ({rom_lines} ROM lines)")


if __name__ == "__main__":
    main()
