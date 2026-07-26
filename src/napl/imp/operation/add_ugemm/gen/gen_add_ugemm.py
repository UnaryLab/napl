import math
from pathlib import Path

import torch

from napl.sim.operation import add_ugemm


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_ugemm.vec"
PARAMS = ROOT / "vec" / "add_ugemm_params.vh"
ENTRY = 8
COUNT_WIDTH = math.ceil(math.log2(ENTRY + 1))
SEGMENT = list(range(1 << ENTRY))
ACC_WIDTH = math.ceil(math.log2(2 * ENTRY * len(SEGMENT) + 1)) + 1


def bits(value):
    return [(value >> index) & 1 for index in range(ENTRY)]


def bus(value):
    return f"{value:0{ENTRY}b}"


def main():
    configs = (
        {"polarity": "unipolar", "scaled": True},
        {"polarity": "bipolar", "scaled": True},
        {"polarity": "unipolar", "scaled": False},
        {"polarity": "bipolar", "scaled": False},
    )
    models = tuple(add_ugemm(dict(config)) for config in configs)
    rows = SEGMENT + SEGMENT
    reset_at = len(SEGMENT)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALED {int(configs[0]['scaled'])}\n"
        f"`define GEN_UNSCALED {int(configs[2]['scaled'])}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_COUNT_WIDTH {COUNT_WIDTH}\n"
        f"`define GEN_ACC_WIDTH {ACC_WIDTH}\n"
        f"`define GEN_PP_DELAY {models[0].hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for index, value in enumerate(rows):
            reset = int(index == 0 or index == reset_at)
            if reset:
                for model in models:
                    model.reset()

            values = (
                value,
                value ^ 0xA5,
                (value * 37) & 0xFF,
                (value * 73 + 19) & 0xFF,
            )
            results = []
            for model, pattern in zip(models, values):
                input_spikes = torch.tensor(bits(pattern), dtype=model.stype)
                results.append(int(model(input_spikes, dim=-1).item()))

            output.write(
                f"{reset} "
                + " ".join(
                    f"{bus(pattern)} {result}"
                    for pattern, result in zip(values, results)
                )
                + "\n"
            )

    print(
        f"wrote {VEC} ({len(rows)} vectors, reset@{reset_at}) and "
        f"{PARAMS} (ENTRY={ENTRY}, ACC_WIDTH={ACC_WIDTH})"
    )


if __name__ == "__main__":
    main()
