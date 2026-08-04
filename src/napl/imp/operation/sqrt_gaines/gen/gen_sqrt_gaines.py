import sys
from pathlib import Path

import torch

from napl.sim.operation import sqrt_gaines


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import require_seeded_sys, encode_value, rep_values


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "sqrt_gaines.vec"
PARAMS = ROOT / "vec" / "sqrt_gaines_params.vh"
ROM = ROOT / "vec" / "sqrt_gaines_rom.hex"
WIDTH = 5
BASE_CONFIG = {
    "width": WIDTH,
    "generator": "sobol",
    "dim": 4,
}
CODEC_UNI = {
    "polarity": "unipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}
CODEC_BI = {**CODEC_UNI, "polarity": "bipolar"}


require_seeded_sys(BASE_CONFIG, CODEC_UNI, CODEC_BI)


def build_stream(codec):
    stream = []
    for value in rep_values("unipolar", value_range=(0.0, 1.0)):
        stream.extend(encode_value(codec, value))
    return stream


def drive(model, stream, reset_at):
    outputs = []
    for index, bit in enumerate(stream):
        if index == reset_at:
            model.reset()
        input_spike = torch.tensor([bit], dtype=model.stype)
        outputs.append(int(model(input_spike).item()))
    return outputs


def main():
    stream_uni = build_stream(CODEC_UNI)
    stream_bi = build_stream(CODEC_BI)
    assert len(stream_uni) == len(stream_bi)
    reset_at = len(stream_uni) // 2

    model_uni = sqrt_gaines({"polarity": "unipolar", **BASE_CONFIG})
    model_bi = sqrt_gaines({"polarity": "bipolar", **BASE_CONFIG})
    model_uni.reset()
    model_bi.reset()
    out_uni = drive(model_uni, stream_uni, reset_at)
    out_bi = drive(model_bi, stream_bi, reset_at)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_text(
        "\n".join(
            f"{int(value.item()):0{WIDTH}b}"
            for value in model_uni.rand_seq
        )
        + "\n"
    )
    PARAMS.write_text(
        f"`define GEN_WIDTH {WIDTH}\n"
        f"`define GEN_PP_DELAY {model_uni.hw.pp_delay}\n"
    )
    with VEC.open("w") as output:
        for index, row in enumerate(
            zip(stream_uni, out_uni, stream_bi, out_bi)
        ):
            reset_flag = int(index == 0 or index == reset_at)
            output.write(
                f"{reset_flag} " + " ".join(str(value) for value in row) + "\n"
            )

    print(
        f"wrote {VEC} ({len(stream_uni)} vectors, reset@{reset_at}), "
        f"{PARAMS}, and {ROM} ({model_uni.rand_seq.numel()} ROM lines)"
    )


if __name__ == "__main__":
    main()
