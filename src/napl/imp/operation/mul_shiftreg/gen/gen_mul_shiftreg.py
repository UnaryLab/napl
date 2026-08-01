"""Generate mul_shiftreg vectors and its model-derived threshold ROM."""
import sys
from pathlib import Path

import torch

from napl.sim.operation import mul_shiftreg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "mul_shiftreg.vec"
PARAMS = ROOT / "vec" / "mul_shiftreg_params.vh"
ROM = ROOT / "vec" / "mul_shiftreg_rom.hex"

# Configuration mirrors test_mul_shiftreg.py.
WIDTH = 4
TIMESTEPS = 256
OP_CONFIG = {"width": WIDTH, "generator": "sobol", "dim": 1}


def build_streams(polarity):
    codec_0 = {
        "polarity": polarity,
        "timestep": TIMESTEPS,
        "generator": "sobol",
        "dim": 1,
    }
    codec_1 = dict(codec_0, dim=2)
    return pair_streams(codec_0, codec_1, rep_pairs(polarity, polarity))


def drive(model, input_0, input_1):
    result = model(
        torch.tensor(input_0, dtype=model.stype),
        torch.tensor(input_1, dtype=model.stype),
    )
    return int(result.item())


def main():
    streams = {
        polarity: build_streams(polarity)
        for polarity in ("unipolar", "bipolar")
    }
    assert len(streams["unipolar"][0]) == len(streams["bipolar"][0])
    assert len(streams["unipolar"][1]) == len(streams["bipolar"][1])

    models = {
        polarity: mul_shiftreg(dict(OP_CONFIG, polarity=polarity))
        for polarity in ("unipolar", "bipolar")
    }
    for model in models.values():
        model.reset()

    rng_values = models["unipolar"].rng_seq.detach().reshape(-1)
    assert len(rng_values) == 2**WIDTH
    ROM.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_text(
        "".join(f"{int(value.item()):0{WIDTH}b}\n" for value in rng_values)
    )
    PARAMS.write_text(
        f"`define GEN_WIDTH {WIDTH}\n"
        f"`define GEN_PP_DELAY {models['unipolar'].hw.pp_delay}\n"
    )

    stream_u = streams["unipolar"]
    stream_b = streams["bipolar"]
    reset_at = len(stream_u[0]) // 2
    rows = 0
    with VEC.open("w") as output:
        for index, (in_0_u, in_1_u, in_0_b, in_1_b) in enumerate(
            zip(stream_u[0], stream_u[1], stream_b[0], stream_b[1])
        ):
            reset_flag = int(index == 0 or index == reset_at)
            if reset_flag:
                for model in models.values():
                    model.reset()
            out_u = drive(models["unipolar"], in_0_u, in_1_u)
            out_b = drive(models["bipolar"], in_0_b, in_1_b)
            output.write(
                f"{reset_flag} {in_0_u} {in_1_u} {out_u} "
                f"{in_0_b} {in_1_b} {out_b}\n"
            )
            rows += 1

    print(
        f"wrote {VEC} ({rows} vectors, reset@0/{reset_at}), "
        f"{PARAMS} and {ROM} (WIDTH={WIDTH})"
    )


if __name__ == "__main__":
    main()
