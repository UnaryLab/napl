"""
Generate golden vectors for the subabs RTL module from napl's Python model
(napl.sim.operation.subabs), so the testbench checks the Verilog against the
actual simulator rather than a hand-derived truth table.

Output: ../vec/subabs.vec, one line per input combination:

    <input_0> <input_1> <output>      (each 0/1, space-separated)

subabs is stateless and takes two spikes, so the exhaustive sweep of its two
1-bit inputs (4 rows) fully characterizes the circuit; the correlated streams
test_subabs.py drives are sequences of those same four symbols.

Run inside the `napl` conda env:
    python gen/gen_subabs.py
"""
import itertools
from pathlib import Path

import torch

from napl.sim.operation import subabs


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "subabs.vec"
PARAMS = ROOT / "vec" / "subabs_params.vh"

# Mirrors test_subabs.py: subabs is unipolar only.
CONFIG = {"polarity": "unipolar"}


def main():
    model = subabs(dict(CONFIG))
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_PP_DELAY {model.hw.pp_delay}\n")

    rows = 0
    with VEC.open("w") as output:
        for bit_0, bit_1 in itertools.product((0, 1), repeat=2):
            spikes = (torch.tensor(bit_0, dtype=model.stype),
                      torch.tensor(bit_1, dtype=model.stype))
            result = int(model(*spikes).item())
            output.write(f"{bit_0} {bit_1} {result}\n")
            rows += 1

    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} "
          f"(GEN_PP_DELAY={model.hw.pp_delay})")


if __name__ == "__main__":
    main()
