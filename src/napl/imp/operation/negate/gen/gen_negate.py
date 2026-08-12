"""
Generate golden vectors for the negate RTL module from napl's Python model
(napl.sim.operation.negate), so the testbench checks the Verilog against the
actual simulator rather than a hand-derived truth table.

Output: ../vec/negate.vec, one line per input value:

    <input> <output>      (each 0/1, space-separated)

negate is stateless and takes a single spike, so the exhaustive sweep of its one
input (2 rows) fully characterizes the circuit. The suite-scale stream that
test_negate.py drives is the same two symbols in a different order.

Run inside the `napl` conda env:
    python gen/gen_negate.py
"""
from pathlib import Path

import torch

from napl.sim.operation import negate


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "negate.vec"
PARAMS = ROOT / "vec" / "negate_params.vh"

# Mirrors test_negate.py: negate is bipolar only.
CONFIG = {"polarity": "bipolar"}


def main():
    model = negate(dict(CONFIG))
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_PP_DELAY {model.hw.pp_delay}\n")

    rows = 0
    with VEC.open("w") as output:
        for bit in (0, 1):
            result = int(model(torch.tensor(bit, dtype=model.stype)).item())
            output.write(f"{bit} {result}\n")
            rows += 1

    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} "
          f"(GEN_PP_DELAY={model.hw.pp_delay})")


if __name__ == "__main__":
    main()
