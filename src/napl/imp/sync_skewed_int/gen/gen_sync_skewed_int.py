import sys
from pathlib import Path

import torch

from napl.sim.operation import sync_skewed_int


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "sync_skewed_int.vec"
PARAMS = ROOT / "vec" / "sync_skewed_int_params.vh"
CONFIG = {"width": 4}
CODEC_1 = {
    "polarity": "unipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}
CODEC_2 = {
    "polarity": "unipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 3,
}


def main():
    pairs = rep_pairs("unipolar", "unipolar")
    stream_1, stream_2 = pair_streams(CODEC_1, CODEC_2, pairs)
    model = sync_skewed_int(dict(CONFIG))
    model.reset()
    reset_at = len(stream_1) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_WIDTH {CONFIG['width']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for index, (bit_1, bit_2) in enumerate(zip(stream_1, stream_2)):
            reset_flag = int(index == 0 or index == reset_at)
            if index == reset_at:
                model.reset()
            in_1 = torch.tensor([bit_1], dtype=model.stype)
            in_2 = torch.tensor([bit_2], dtype=model.stype)
            out_1, out_2 = model(in_1, in_2)
            output.write(
                f"{reset_flag} {bit_1} {bit_2} "
                f"{int(out_1.item())} {int(out_2.item())}\n"
            )

    print(
        f"wrote {VEC} ({len(stream_1)} vectors, reset@{reset_at}) "
        f"and {PARAMS}"
    )


if __name__ == "__main__":
    main()
