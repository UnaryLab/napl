#!/usr/bin/env python
"""Emit golden vectors for mul_csg from the napl Python model.

mul_csg is a stateful bit-serial op: a fixed-point operand i_in_1 is compared
against a generator ROM num_seq[idx], with idx counters advanced by the input
spike. The RTL inherits its size (WIDTH = ceil(log2(timestep))) from this config,
and the num_seq ROM is GENERATED here from the model and loaded by the RTL via
$readmemb, so the table follows WIDTH. We pin
to the TEST config (tests/operation/test_mul_csg.py: timestep=1024, generator
'sobol') -> WIDTH=10, LEN=1024, so the RTL is validated at the size the test runs.
Expected outputs come from the napl model, never a hand truth table.

Per cycle we record:  rst in_0 in_1u out_uni in_1b out_bi
  rst    -- 1 on the first cycle of each independent sequence (pulse i_rst_n low)
  in_0   -- the 1-bit input spike for this cycle
  in_1u  -- unipolar operand:  round(in_1 * LEN)              (drives _unipolar)
  out_uni-- unipolar model output
  in_1b  -- bipolar operand:   round((in_1 + 1)/2 * LEN)      (drives _bipolar)
  out_bi -- bipolar model output

in_1u/in_1b are constant within a sequence (i_in_1 is a held operand), but the
columns carry them every cycle so the testbench can drive without parsing state.
"""
import math
import os
import sys

import torch

from napl.sim.operation import mul_csg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from _gen_common import encode_value

# The op's sizing config, read ONCE here, mirrored from the TEST (the source of
# truth). mul_csg's only numeric size key is 'timestep'; WIDTH = ceil(log2(
# timestep)) drives every RTL bus width (counter width, ROM value width, operand
# bus = WIDTH+1) and the ROM depth (LEN = 2**WIDTH). WIDTH is the single source of
# truth: it builds the model AND is emitted as `GEN_WIDTH so the testbench
# overrides the RTL parameter with the same value -> sim and RTL cannot drift. The
# num_seq ROM is generated below from a real model instance at this config, so the
# RTL's $readmemb-loaded table is the model's actual sequence at the inherited
# WIDTH.
TIMESTEP = 1024  # mirrors tests/operation/test_mul_csg.py codec_config['timestep']
WIDTH = math.ceil(math.log2(TIMESTEP))
LEN = 2 ** WIDTH
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vec")
OUT_PATH = os.path.join(OUT_DIR, "mul_csg.vec")
PARAMS_PATH = os.path.join(OUT_DIR, "mul_csg_params.vh")
ROM_PATH = os.path.join(OUT_DIR, "mul_csg_rom.hex")

# test_mul_csg.py encodes input_0 with a bipolar sobol encoder at timestep=1024;
# encode the in_0 spike stream with the same sobol family at len=LEN to stay
# bit-identical with the streams the test feeds the op.
CODEC_IN0 = {"polarity": "bipolar", "timestep": LEN, "generator": "sobol", "dim": 1}

CFG_UNI = {"polarity": "unipolar", "timestep": LEN, "generator": "sobol"}
CFG_BI = {"polarity": "bipolar", "timestep": LEN, "generator": "sobol"}


def _drive(model, stream, in1_value):
    """Drive a reset model with the spike stream and held operand; return outputs."""
    in1 = torch.tensor([in1_value], dtype=torch.float32)
    out = []
    for s in stream:
        o = model(torch.tensor([s], dtype=torch.int8), in1)
        out.append(int(o.reshape(-1)[0].item()))
    return out


def run_seq(in0_stream, in1_value):
    """Return (out_uni, out_bi) lists for one operand value over the spike stream.

    Each polarity is a fresh, reset model. in1_value is the real-valued operand;
    unipolar interprets it in [0,1], bipolar in [-1,1].
    """
    mu = mul_csg(CFG_UNI)
    mu.reset()
    out_uni = _drive(mu, in0_stream, in1_value)

    mb = mul_csg(CFG_BI)
    mb.reset()
    out_bi = _drive(mb, in0_stream, in1_value)

    return out_uni, out_bi


def run_seq_midreset(pre_stream, pre_val, post_stream, post_val):
    """Stateful reset-equivalence: dirty the counters with pre_stream/pre_val,
    call reset() mid-stream, then continue with post_stream/post_val. The
    post-reset half must equal a FRESH reset model run on the same post inputs
    (proved by the testbench pulsing i_rst_n on the rst=1 row that opens it).
    Returns (out_uni, out_bi) for the post-reset half only.
    """
    mu = mul_csg(CFG_UNI)
    mu.reset()
    _drive(mu, pre_stream, pre_val)   # dirty seq_idx
    mu.reset()                        # mid-stream reset
    out_uni = _drive(mu, post_stream, post_val)

    mb = mul_csg(CFG_BI)
    mb.reset()
    _drive(mb, pre_stream, pre_val)
    mb.reset()
    out_bi = _drive(mb, post_stream, post_val)

    return out_uni, out_bi


def operand_codes(in1_value):
    """Return (in1u, in1b) integer operands, guarding the 1/LEN-grid contract."""
    prob_u = in1_value
    prob_b = (in1_value + 1.0) / 2.0
    in1u = round(prob_u * LEN)
    in1b = round(prob_b * LEN)
    assert abs(prob_u * LEN - in1u) < 1e-9, f"unipolar operand {in1_value} off-grid"
    assert abs(prob_b * LEN - in1b) < 1e-9, f"bipolar operand {in1_value} off-grid"
    return in1u, in1b


def emit_block(lines, in0_stream, in1_value, out_uni, out_bi):
    """Append one sequence: rst=1 on the first cycle, rst=0 after."""
    in1u, in1b = operand_codes(in1_value)
    for t in range(len(in0_stream)):
        rst = 1 if t == 0 else 0
        lines.append(f"{rst} {in0_stream[t]} {in1u} {out_uni[t]} {in1b} {out_bi[t]}")


def write_rom():
    """Emit the num_seq ROM hex from a real model instance at the test config.

    The RTL $readmemb-loads exactly LEN lines, line i = round(num_seq[i]*LEN) as a
    WIDTH-bit binary string. num_seq is read from a built mul_csg (the model's
    actual sequence), not re-derived. Bipolar reuses the SAME table at two indices,
    so one file serves both read ports. The grid assert guarantees the integer
    compare in RTL is bit-exact with the model's float gt (num_seq*LEN is integral).
    """
    m = mul_csg(CFG_UNI)  # unipolar/bipolar share the same (generator, WIDTH) num_seq
    ns = m.num_seq.detach().float().reshape(-1)
    assert ns.numel() == LEN, f"num_seq length {ns.numel()} != LEN {LEN}"
    rom_lines = []
    for i in range(LEN):
        scaled = ns[i].item() * LEN
        v = round(scaled)
        assert abs(scaled - v) < 1e-9, f"num_seq[{i}]={ns[i].item()} off the 1/{LEN} grid"
        assert 0 <= v < LEN, f"num_seq code {v} out of [0,{LEN})"
        rom_lines.append(f"{v:0{WIDTH}b}")
    with open(ROM_PATH, "w") as f:
        f.write("\n".join(rom_lines) + "\n")
    return len(rom_lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Emit the sizing-param header the testbench includes to override the RTL
    # parameter. WIDTH is the single source of truth, mirrored from the test.
    with open(PARAMS_PATH, "w") as f:
        f.write(f"`define GEN_WIDTH {WIDTH}\n")

    # Emit the generated num_seq ROM the RTL $readmemb-loads (follows WIDTH).
    rom_count = write_rom()

    lines = ["rst in_0 in_1u out_uni in_1b out_bi"]

    # The RTL operand i_in_1 is the integer round(prob*LEN); the comparison is
    # bit-exact with the model's float gt() only when prob is an exact multiple of
    # 1/LEN. The same real operand drives BOTH polarity modules, so it must be
    # exact in both interpretations: unipolar prob = value, bipolar prob =
    # (value+1)/2. Picking value = j/512 (j in [0,512], value in [0,1]) makes both
    # probs exact multiples of 1/1024. This is the same width-bit quantization the
    # napl test feeds via gen_rand_tensor; off-grid operands are out of contract.
    operands = [0.0, 1.0 / 512, 0.25, 0.5, 320.0 / 512, 0.75, 511.0 / 512, 1.0]
    # in_0 spike stream comes from the test's bipolar sobol encoder; pair each
    # held operand with a representative in_0 value spanning the bipolar range.
    in0_values = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    for in1_value, in0_value in zip(operands, in0_values):
        in0_stream = encode_value(CODEC_IN0, in0_value)
        out_uni, out_bi = run_seq(in0_stream, in1_value)
        emit_block(lines, in0_stream, in1_value, out_uni, out_bi)

    # Stateful reset-equivalence: dirty the counters on a full pre-stream, then
    # reset() mid-stream and replay a post-stream. The emitted block is the
    # post-reset half opened with rst=1, so the testbench pulses i_rst_n there;
    # it must match a fresh reset() model. Proves the RTL async reset reproduces
    # reset() from a DIRTIED counter state, not just from power-on.
    pre_stream = encode_value(CODEC_IN0, 0.5)    # walks the counters off zero
    post_val = 320.0 / 512                          # representative operand
    post_stream = encode_value(CODEC_IN0, -0.25)   # representative in_0 stream
    out_uni, out_bi = run_seq_midreset(pre_stream, 0.25, post_stream, post_val)
    emit_block(lines, post_stream, post_val, out_uni, out_bi)

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {OUT_PATH} ({len(lines) - 1} vectors), {PARAMS_PATH} "
          f"(GEN_WIDTH={WIDTH}), and {ROM_PATH} ({rom_count} ROM lines)")


if __name__ == "__main__":
    main()
