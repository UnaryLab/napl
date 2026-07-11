"""Emit golden vectors for max_rc from the napl Python model.

Drives the model with random {0,1} spike streams on (input_0, input_1) and
records per cycle: i_in_0 i_in_1 o_max o_arg. max_rc has no polarity branch, so
there is a single output column pair. Run from t=0 after model.reset().
"""
import os
import sys

import torch

from napl.operation import max_rc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from _gen_common import pair_streams, rep_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, os.pardir, "vec", "max_rc.vec")

# test_max_rc.py codec_config1/2: the two encoders feeding max_rc, distinct dims.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def main():
    model = max_rc({"polarity": "bipolar", "timestep": 256})
    model.reset()

    # the test's encoder streams for representative operand pairs, concatenated.
    s0, s1 = pair_streams(CODEC0, CODEC1, rep_pairs("bipolar", "bipolar"))

    os.makedirs(os.path.dirname(VEC), exist_ok=True)
    with open(VEC, "w") as f:
        f.write("# i_in_0 i_in_1 o_max o_arg\n")
        for i0, i1 in zip(s0, s1):
            o_max, o_arg = model(
                torch.tensor([i0], dtype=torch.int8),
                torch.tensor([i1], dtype=torch.int8),
            )
            f.write(f"{i0} {i1} {int(o_max.item())} {int(o_arg.item())}\n")


if __name__ == "__main__":
    main()
