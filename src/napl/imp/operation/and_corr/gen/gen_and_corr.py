from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.operation import and_corr, encode


VEC = Path(__file__).resolve().parent.parent / "vec" / "and_corr.vec"

# Mirror tests/operation/test_and_corr.py: unipolar streams over the full [0, 1]
# range, both operands encoded on the SAME Sobol dim (encoder_dims [1, 1]) so
# they are maximally positively correlated and the AND output is min(a, b).
TIMESTEPS = 256
DIM = 1


def main():
    # Fidelity-scale operand pairs spanning the legal unipolar range. A roll
    # decorrelates the two operand values while both streams stay on Sobol dim 1,
    # matching make_values in the test.
    values_0 = torch.linspace(0.0, 1.0, 64, dtype=global_config.ntype)
    values_1 = values_0.roll(21)

    model = and_corr({"polarity": "unipolar"})
    enc_0 = encode({"polarity": "unipolar", "timestep": TIMESTEPS, "generator": "sobol", "dim": DIM})
    enc_1 = encode({"polarity": "unipolar", "timestep": TIMESTEPS, "generator": "sobol", "dim": DIM})

    count = 0
    VEC.parent.mkdir(parents=True, exist_ok=True)
    with VEC.open("w") as output:
        for _ in range(TIMESTEPS):
            spike_0 = enc_0(values_0)
            spike_1 = enc_1(values_1)
            out = model(spike_0, spike_1)
            for a, b, o in zip(spike_0.tolist(), spike_1.tolist(), out.tolist()):
                output.write(f"{int(a)} {int(b)} {int(o)}\n")
                count += 1

    print(f"wrote {VEC} ({count} vectors)")


if __name__ == "__main__":
    main()
