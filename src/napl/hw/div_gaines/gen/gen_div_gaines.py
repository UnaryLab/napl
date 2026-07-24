import sys
from pathlib import Path

import torch

from napl.sim.operation import div_gaines


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "div_gaines.vec"
PARAMS = ROOT / "vec" / "div_gaines_params.vh"
ROM = ROOT / "vec" / "div_gaines_rom.hex"
DEPTH = 5
BASE_CONFIG = {
    "depth": DEPTH,
    "generator": "sobol",
    "dim": 3,
}
CODEC_UNI_0 = {
    "polarity": "unipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}
CODEC_UNI_1 = {**CODEC_UNI_0, "dim": 2}
CODEC_BI_0 = {**CODEC_UNI_0, "polarity": "bipolar"}
CODEC_BI_1 = {**CODEC_BI_0, "dim": 2}


def drive(model, stream_0, stream_1, reset_at):
    outputs = []
    for index, (bit_0, bit_1) in enumerate(zip(stream_0, stream_1)):
        if index == reset_at:
            model.reset()
        in_0 = torch.tensor([bit_0], dtype=model.stype)
        in_1 = torch.tensor([bit_1], dtype=model.stype)
        outputs.append(int(model(in_0, in_1).item()))
    return outputs


def write_rom(model):
    lines = [
        f"{int(value):0{DEPTH}b}"
        for value in model.rng_seq
    ]
    ROM.write_text("\n".join(lines) + "\n")


def main():
    pairs_uni = rep_pairs("unipolar", "unipolar")
    pairs_bi = rep_pairs("bipolar", "bipolar")
    uni_0, uni_1 = pair_streams(CODEC_UNI_0, CODEC_UNI_1, pairs_uni)
    bi_0, bi_1 = pair_streams(CODEC_BI_0, CODEC_BI_1, pairs_bi)
    assert len(uni_0) == len(bi_0)
    reset_at = len(uni_0) // 2

    model_uni = div_gaines({"polarity": "unipolar", **BASE_CONFIG})
    model_bi = div_gaines({"polarity": "bipolar", **BASE_CONFIG})
    model_uni.reset()
    model_bi.reset()
    out_uni = drive(model_uni, uni_0, uni_1, reset_at)
    out_bi = drive(model_bi, bi_0, bi_1, reset_at)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    write_rom(model_uni)
    PARAMS.write_text(
        f"`define GEN_DEPTH {DEPTH}\n"
        f"`define GEN_PP_DELAY {model_uni.hw.pp_delay}\n"
    )
    with VEC.open("w") as output:
        for index, row in enumerate(
            zip(uni_0, uni_1, out_uni, bi_0, bi_1, out_bi)
        ):
            reset_flag = int(index == 0 or index == reset_at)
            output.write(
                f"{reset_flag} " + " ".join(str(value) for value in row) + "\n"
            )

    print(
        f"wrote {VEC} ({len(uni_0)} vectors, reset@{reset_at}), "
        f"{PARAMS}, and {ROM} ({len(model_uni.rng_seq)} ROM lines)"
    )


if __name__ == "__main__":
    main()
