"""
Generate golden vectors for the mul_unibi_mux RTL module from napl's Python model
(napl.sim.operation.mul_unibi_mux), so the testbench checks the Verilog against
the actual simulator rather than a hand-derived truth table.

Output: ../vec/mul_unibi_mux.vec, one line per timestep:

    <rst> <input_u> <input_b> <output>      (each 0/1, space-separated)

The stimulus has two parts. A transition segment enumerates every
(input_u, input_b, toggle state) combination, which fully characterizes the
state machine. The fidelity segments then encode an operand grid spanning the
full legal range of both operands, unipolar [0, 1] on input_u and bipolar
[-1, 1] on input_b, with the encoder configuration test_mul_unibi_mux.py uses:
distinct Sobol dimensions so the two streams stay decorrelated. Each segment is
an independent run of the one scalar circuit, so every segment starts with a
reset, which also restarts the toggle at the parity the model resets to.

Run inside the `napl` conda env:
    python gen/gen_mul_unibi_mux.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.operation import mul_unibi_mux

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "mul_unibi_mux.vec"
PARAMS = ROOT / "vec" / "mul_unibi_mux_params.vh"

TIMESTEP = 256
# Encoder configurations mirror test_mul_unibi_mux.py: the operand polarities
# come from polarity_io and the streaming suite gives each operand its own Sobol
# dimension, which is what keeps the two streams decorrelated.
CODEC_U = {"polarity": "unipolar", "timestep": TIMESTEP, "generator": "sobol", "dim": 1}
CODEC_B = {"polarity": "bipolar", "timestep": TIMESTEP, "generator": "sobol", "dim": 2}
# Operand grid over the full legal range of both operands.
GRID = 8
# Every (input_u, input_b, toggle state) combination: the toggle alternates on
# every timestep, so holding each operand pair for two timesteps visits both
# states of it.
TRANSITIONS = [(0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (1, 0), (1, 1), (1, 1)]


def grid_segments():
    """Encoded (input_u, input_b) stream pairs over the operand grid."""
    segments = []
    for value_u in torch.linspace(0.0, 1.0, GRID).tolist():
        for value_b in torch.linspace(-1.0, 1.0, GRID).tolist():
            segments.append(list(zip(encode_value(CODEC_U, value_u),
                                     encode_value(CODEC_B, value_b))))
    return segments


def main():
    model = mul_unibi_mux()
    segments = [TRANSITIONS] + grid_segments()

    covered = set()
    rows = []
    for segment in segments:
        model.reset()
        for index, (spike_u, spike_b) in enumerate(segment):
            state = int(model.state.item())
            result = int(model(torch.tensor(spike_u, dtype=model.stype),
                               torch.tensor(spike_b, dtype=model.stype)).item())
            rows.append((int(index == 0), spike_u, spike_b, result))
            covered.add((spike_u, spike_b, state))
    assert len(covered) == 8, \
        f"the stimulus covers {len(covered)} of the 8 (input_u, input_b, state) combinations"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_PP_DELAY {model.hw.pp_delay}\n")
    with VEC.open("w") as output:
        for reset, spike_u, spike_b, result in rows:
            output.write(f"{reset} {spike_u} {spike_b} {result}\n")

    print(f"wrote {VEC} ({len(rows)} vectors, {len(segments)} reset segments) "
          f"and {PARAMS} (GEN_PP_DELAY={model.hw.pp_delay})")


if __name__ == "__main__":
    main()
