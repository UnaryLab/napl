#!/usr/bin/env python
"""Emit golden vectors for mul_ugemm from the napl Python model.

mul_ugemm is a stateful bit-serial op: a fixed-point operand i_input_1 is compared
against the number sequence num_seq[idx], with idx advanced by the input spike.
The RTL inherits its size (WIDTH = ceil(log2(timestep))) from this config and
produces num_seq online with a `sobol` generator, so this script emits the
WIDTH-entry direction-vector table that generator reads instead of a full-period
ROM. The vectors are pinned to timestep=1024 with the 'sobol' generator ->
WIDTH=10, LEN=1024, which is the size the RTL is verified at. Expected outputs
come from the napl model, never a hand truth table.

Sequence-index budget: the model advances seq_idx without a modulo, so a block
holds at most LEN enabling timesteps and every timestep after the LENth raises
IndexError rather than reading a wrapped sample. The blocks below sit at that
ceiling, so the last compared row reads index LEN-1 and no row compares the
generator's wrap. mul_ugemm_regen, whose rng index wraps modulo, is where the
co-simulation covers the wrapped sample of the shared sobol circuit.

Per cycle we record:  rst in_0 in_1u out_uni in_1b out_bi
  rst    -- 1 on the first cycle of each independent sequence (pulse i_rst_n low)
  in_0   -- the 1-bit input spike for this cycle
  in_1u  -- unipolar operand:  round(in_1 * LEN)              (drives _unipolar)
  out_uni-- unipolar model output
  in_1b  -- bipolar operand:   round((in_1 + 1)/2 * LEN)      (drives _bipolar)
  out_bi -- bipolar model output

in_1u/in_1b are constant within a sequence (i_input_1 is a held operand), but the
columns carry them every cycle so the testbench can drive without parsing state.
"""
import math
import os
import sys

import torch

from napl.sim.operation import mul_ugemm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from _gen_common import encode_value

# WIDTH derives from the model timestep and sizes the sequence, operands, and the
# direction-vector table.
TIMESTEP = 1024  # timestep of the verified golden vectors
WIDTH = math.ceil(math.log2(TIMESTEP))
LEN = 2 ** WIDTH
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vec")
OUT_PATH = os.path.join(OUT_DIR, "mul_ugemm.vec")
PARAMS_PATH = os.path.join(OUT_DIR, "mul_ugemm_params.vh")
DIRVEC_PATH = os.path.join(OUT_DIR, "mul_ugemm_dv.hex")

# input_0 uses a bipolar Sobol encoder.
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
    mu = mul_ugemm(CFG_UNI)
    mu.reset()
    out_uni = _drive(mu, in0_stream, in1_value)

    mb = mul_ugemm(CFG_BI)
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
    mu = mul_ugemm(CFG_UNI)
    mu.reset()
    _drive(mu, pre_stream, pre_val)   # dirty seq_idx
    mu.reset()                        # mid-stream reset
    out_uni = _drive(mu, post_stream, post_val)

    mb = mul_ugemm(CFG_BI)
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
    assert values.numel() == period, f'{name} holds {values.numel()} points, not {period}'
    codes = []
    for index in range(period):
        scaled = values[index].item() * period
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f'{name}[{index}] is off the 1/{period} grid'
        codes.append(code)
    assert codes[0] == 0, f'{name} starts at {codes[0]}, not the post-reset 0'

    vectors = [codes[2 ** k] ^ codes[2 ** k - 1] for k in range(width)]
    state = 0
    for index in range(period):
        assert state == codes[index], \
            f'{name} is not a gray-code Sobol sequence: the recurrence gives {state} ' \
            f'at index {index}, the model gives {codes[index]}'
        # The all-ones counter state takes the top position, which returns to 0.
        position = width - 1 if index == period - 1 else (~index & (index + 1)).bit_length() - 1
        state ^= vectors[position]
    assert state == 0, f'{name} returns to {state} on the wrap, not 0'
    return [f'{vector:0{width}b}' for vector in vectors]


def write_dirvec():
    """Emit the direction-vector table the RTL sobol generators read.

    The table is recovered from a built mul_ugemm's own num_seq. Both polarities
    share one (generator, WIDTH) sequence and bipolar walks it at two indices, so
    a single file serves every sobol instance.
    """
    uni = mul_ugemm(CFG_UNI).num_seq
    bi = mul_ugemm(CFG_BI).num_seq
    assert torch.equal(uni, bi), "the two polarities no longer share one num_seq"
    rows = dirvec_rows(uni, WIDTH, "mul_ugemm num_seq")
    with open(DIRVEC_PATH, "w") as f:
        f.write("\n".join(rows) + "\n")
    return len(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dirvec_count = write_dirvec()

    lines = ["rst in_0 in_1u out_uni in_1b out_bi"]

    # Both polarity mappings must land on the 1/LEN fixed-point grid.
    operands = [0.0, 1.0 / 512, 0.25, 0.5, 320.0 / 512, 0.75, 511.0 / 512, 1.0]
    in0_values = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

    for in1_value, in0_value in zip(operands, in0_values):
        in0_stream = encode_value(CODEC_IN0, in0_value)
        out_uni, out_bi = run_seq(in0_stream, in1_value)
        emit_block(lines, in0_stream, in1_value, out_uni, out_bi)

    # rst marks the first cycle after resetting both sequence counters.
    pre_stream = encode_value(CODEC_IN0, 0.5)    # walks the counters off zero
    post_val = 320.0 / 512                          # representative operand
    post_stream = encode_value(CODEC_IN0, -0.25)   # representative in_0 stream
    out_uni, out_bi = run_seq_midreset(pre_stream, 0.25, post_stream, post_val)
    emit_block(lines, post_stream, post_val, out_uni, out_bi)

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    pp_delay = mul_ugemm(CFG_UNI).hw.pp_delay
    with open(PARAMS_PATH, "w") as f:
        f.write(f"`define GEN_WIDTH {WIDTH}\n")
        f.write(f"`define GEN_PP_DELAY {pp_delay}\n")
        f.write(f"`define GEN_VECTORS {len(lines) - 1}\n")

    print(f"wrote {OUT_PATH} ({len(lines) - 1} vectors), {PARAMS_PATH} "
          f"(GEN_WIDTH={WIDTH}), and {DIRVEC_PATH} ({dirvec_count} direction vectors)")


if __name__ == "__main__":
    main()
