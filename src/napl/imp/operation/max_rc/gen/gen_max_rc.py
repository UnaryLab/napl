"""Emit golden vectors for max_rc from the napl Python model.

Drives the model with random {0,1} spike streams on (input_0, input_1) and
records per cycle: reset i_input_0 i_input_1 o_max o_arg. max_rc has no polarity
branch. Reset is asserted before the first row and once after state has changed.
"""
import os
import sys

import torch

from napl.sim.operation import max_rc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from _gen_common import pair_streams, rep_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, os.pardir, "vec", "max_rc.vec")
PARAMS = os.path.join(HERE, os.pardir, "vec", "max_rc_params.vh")

# Encoder settings mirror test_max_rc.py and use distinct Sobol dimensions.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def main():
    model = max_rc({"polarity": "bipolar", "timestep": 256})
    model.reset()

    s0, s1 = pair_streams(CODEC0, CODEC1, rep_pairs("bipolar", "bipolar"))
    reset_at = len(s0) // 2

    os.makedirs(os.path.dirname(VEC), exist_ok=True)
    with open(PARAMS, "w") as f:
        f.write(f"`define GEN_PP_DELAY {model.hw.pp_delay}\n")
    with open(VEC, "w") as f:
        for index, (i0, i1) in enumerate(zip(s0, s1)):
            reset = int(index == 0 or index == reset_at)
            if index == reset_at:
                model.reset()
            o_max, o_arg = model(
                torch.tensor([i0], dtype=torch.int8),
                torch.tensor([i1], dtype=torch.int8),
            )
            f.write(
                f"{reset} {i0} {i1} "
                f"{int(o_max.item())} {int(o_arg.item())}\n"
            )
    print(
        f"wrote {VEC} ({len(s0)} vectors, reset@{reset_at}) and {PARAMS} "
        f"(GEN_PP_DELAY={model.hw.pp_delay})"
    )


if __name__ == "__main__":
    main()
