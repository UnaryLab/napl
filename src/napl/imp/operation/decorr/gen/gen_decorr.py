#!/usr/bin/env python
"""Emit golden vectors for the decorr RTL module from the napl Python model.

decorr is stateful: each stream owns a DEPTH-position shuffle buffer whose
per-timestep position comes from an auxiliary number sequence. The RTL inherits
DEPTH and the sequence length from this config, and the two sequences are emitted
here as one ROM word per timestep, {idx_1, idx_0}, which the RTL loads with
$readmemb. Expected outputs come from the model, never a hand truth table.

The stimulus is the test's own encoder configuration, one shared Sobol dimension
for both streams, which is the maximally correlated input regime decorr exists for.

Per cycle we record:  rst in_0 in_1 out_0 out_1
  rst   -- 1 on the first cycle of each block (pulse i_rst_n low there)
  in_0  -- current spike of the first stream
  in_1  -- current spike of the second stream
  out_0 -- first model output
  out_1 -- second model output

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_decorr.py
"""
import math
import os
import sys

import torch

from napl.sim.operation import decorr
from napl.syn import translate_node

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from _gen_common import encode_value

# Sizing and encoder settings mirror test_decorr.py.
CONFIG = {"polarity": "unipolar", "depth": 4, "timestep": 256,
          "generator": "sys", "seed": 7}
CODEC = {"polarity": "unipolar", "timestep": CONFIG["timestep"],
         "generator": "sobol", "dim": 1}

DEPTH = CONFIG["depth"]
SEQ_LEN = CONFIG["timestep"]
IDX_W = max(1, math.ceil(math.log2(DEPTH)))
SEQ_W = max(1, math.ceil(math.log2(SEQ_LEN)))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vec")
OUT_PATH = os.path.join(OUT_DIR, "decorr.vec")
PARAMS_PATH = os.path.join(OUT_DIR, "decorr_params.vh")
ROM_PATH = os.path.join(OUT_DIR, "decorr_rom.hex")

# The last pair repeats the third, so the replay after a reset that follows a
# dirtied buffer must reproduce the earlier block bit for bit.
VALUE_PAIRS = [(0.0, 1.0), (1.0, 0.0), (0.25, 0.75), (0.5, 0.5),
               (0.875, 0.125), (0.25, 0.75)]
REPLAY_OF = 2
# Block LONG_BLOCK repeats its stream, so it outlives one sequence period and is
# the only block whose rows compare the model's sequence-index wrap against the
# RTL's counter wrap.
LONG_BLOCK = 4
LONG_REPEAT = 2


def run_block(model, stream_0, stream_1):
    """Outputs of one reset-to-reset block over the two spike streams."""
    model.reset()
    rows = []
    for bit_0, bit_1 in zip(stream_0, stream_1):
        out_0, out_1 = model(torch.tensor(bit_0, dtype=model.stype),
                             torch.tensor(bit_1, dtype=model.stype))
        rows.append((int(out_0.item()), int(out_1.item())))
    return rows


def write_rom(model):
    """Emit the position ROM, one word per timestep, as {idx_1, idx_0}."""
    seq_0, seq_1 = model.rand_seq_idx
    assert len(seq_0) == len(seq_1) == SEQ_LEN, (len(seq_0), len(seq_1))
    lines = []
    for index_0, index_1 in zip(seq_0, seq_1):
        for index in (index_0, index_1):
            assert 0 <= index < DEPTH, f"position {index} outside [0,{DEPTH})"
        lines.append(f"{index_1:0{IDX_W}b}{index_0:0{IDX_W}b}")
    with open(ROM_PATH, "w") as rom:
        rom.write("\n".join(lines) + "\n")
    return len(lines)


def check_mapping():
    """Require mapping.yaml to resolve the parameters these vectors were built from."""
    binding = translate_node({"class": "decorr", "config": CONFIG})
    assert binding.rtl_module == "decorr", binding.rtl_module
    assert binding.parameters == {"DEPTH": DEPTH, "IDX_W": IDX_W,
                                 "SEQ_LEN": SEQ_LEN, "SEQ_W": SEQ_W}, binding.parameters


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    check_mapping()

    model = decorr(dict(CONFIG))
    # Polarity only reinterprets the rate, so both polarities emit the same bits;
    # that is what lets one bare RTL module cover both.
    model_bipolar = decorr(dict(CONFIG, polarity="bipolar"))

    rom_count = write_rom(model)
    with open(PARAMS_PATH, "w") as params:
        params.write(f"`define GEN_DEPTH {DEPTH}\n"
                     f"`define GEN_IDX_W {IDX_W}\n"
                     f"`define GEN_SEQ_LEN {SEQ_LEN}\n"
                     f"`define GEN_SEQ_W {SEQ_W}\n"
                     f"`define GEN_PP_DELAY {model.hw.pp_delay}\n")

    lines = ["rst in_0 in_1 out_0 out_1"]
    blocks = []
    for block, (value_0, value_1) in enumerate(VALUE_PAIRS):
        repeat = LONG_REPEAT if block == LONG_BLOCK else 1
        stream_0 = encode_value(CODEC, value_0) * repeat
        stream_1 = encode_value(CODEC, value_1) * repeat
        rows = run_block(model, stream_0, stream_1)
        assert rows == run_block(model_bipolar, stream_0, stream_1), \
            f"({value_0}, {value_1}) differs between polarities"
        blocks.append(rows)
        for step, ((bit_0, bit_1), (out_0, out_1)) in enumerate(
                zip(zip(stream_0, stream_1), rows)):
            lines.append(f"{1 if step == 0 else 0} {bit_0} {bit_1} {out_0} {out_1}")

    assert blocks[-1] == blocks[REPLAY_OF], \
        "reset after a dirtied buffer did not reproduce the earlier block"

    with open(OUT_PATH, "w") as vectors:
        vectors.write("\n".join(lines) + "\n")

    print(f"wrote {OUT_PATH} ({len(lines) - 1} vectors, {len(blocks)} reset blocks), "
          f"{PARAMS_PATH} (GEN_DEPTH={DEPTH}, GEN_IDX_W={IDX_W}, GEN_SEQ_LEN={SEQ_LEN}, "
          f"GEN_SEQ_W={SEQ_W}, GEN_PP_DELAY={model.hw.pp_delay}), and "
          f"{ROM_PATH} ({rom_count} ROM words)")


if __name__ == "__main__":
    main()
