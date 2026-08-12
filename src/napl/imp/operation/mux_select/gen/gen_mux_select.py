"""
Generate golden vectors for the mux_select RTL module from napl's Python model
(napl.sim.operation.mux_select), so the testbench checks the Verilog against the
actual simulator rather than a hand-derived truth table.

Output: ../vec/mux_select.vec, one line per input combination:

    <select> <input_a> <input_b> <output>      (each 0/1, space-separated)

mux_select is stateless and its output is the bitwise select `c ? a : b`,
polarity-agnostic on the spikes, so a unipolar and a bipolar model produce the
same output bit. The exhaustive sweep of the three 1-bit inputs (8 rows) fully
characterizes the combinational circuit; both polarity models are driven over it
and asserted to agree before recording the single output column.

Run inside the `napl` conda env:
    python gen/gen_mux_select.py
"""
import itertools
from pathlib import Path

import torch

from napl.sim.operation import mux_select


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "mux_select.vec"
PARAMS = ROOT / "vec" / "mux_select_params.vh"


def main():
    unipolar = mux_select({"polarity": "unipolar"})
    bipolar = mux_select({"polarity": "bipolar"})
    unipolar.reset()
    bipolar.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_PP_DELAY {unipolar.hw.pp_delay}\n")

    rows = 0
    with VEC.open("w") as output:
        for select, input_a, input_b in itertools.product((0, 1), repeat=3):
            spikes = (torch.tensor(select),
                      torch.tensor(input_a),
                      torch.tensor(input_b))
            out_uni = int(unipolar(*spikes).item())
            out_bi = int(bipolar(*spikes).item())
            # The select is bitwise, so both polarities emit the same bit.
            assert out_uni == out_bi, (select, input_a, input_b, out_uni, out_bi)
            output.write(f"{select} {input_a} {input_b} {out_uni}\n")
            rows += 1

    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} "
          f"(GEN_PP_DELAY={unipolar.hw.pp_delay})")


if __name__ == "__main__":
    main()
