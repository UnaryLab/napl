import math
from pathlib import Path

import torch

from napl.sim.operation import add_gaines, encode


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_gaines.vec"
PARAMS = ROOT / "vec" / "add_gaines_params.vh"
ROM = ROOT / "vec" / "add_gaines_rom.hex"
TIMESTEP = 256
SCALED_ENTRY = 8
UNSCALED_ENTRY = 4
SCALED_SELECT_WIDTH = int(math.log2(SCALED_ENTRY))
UNSCALED_SELECT_WIDTH = int(math.log2(UNSCALED_ENTRY))
SCALED = {
    "scaled": True,
    "entry": SCALED_ENTRY,
    "generator": "sobol",
    "dim": 5,
}
UNSCALED = {"scaled": False}
SCALED_CODEC = {"timestep": TIMESTEP, "generator": "sobol", "dim": 1}
UNSCALED_CODECS = [
    {"polarity": "unipolar", "timestep": TIMESTEP, "generator": "sobol", "dim": dim}
    for dim in range(1, UNSCALED_ENTRY + 1)
]
# Rail rates driving each input stream to the codec's extremes (all-zeros, the
# rate-0.5 bipolar-zero midpoint, all-ones), appended to every arm so
# build_segments zips a matching count.
CORNER_RATES = (0.0, 0.5, 1.0)


def corner_values(polarity):
    return [rate if polarity == "unipolar" else 2 * rate - 1 for rate in CORNER_RATES]


def scaled_streams(polarity):
    lo = -0.75 if polarity == "bipolar" else 0.0
    sweep = torch.cat((torch.linspace(lo, 0.75, 64), torch.tensor(corner_values(polarity))))
    values = sweep.repeat(SCALED_ENTRY, 1)
    enc = encode({"polarity": polarity, **SCALED_CODEC})
    enc.reset()
    spikes = torch.stack([enc(values) for _ in range(TIMESTEP)])
    return spikes.permute(2, 0, 1)


def unscaled_streams():
    base = torch.linspace(0.0, 0.15, 64)
    rolled = torch.stack([base.roll(index * 7) for index in range(UNSCALED_ENTRY)])
    corner = torch.tensor(corner_values("unipolar")).repeat(UNSCALED_ENTRY, 1)
    values = torch.cat((rolled, corner), dim=1)
    encoders = [encode(config) for config in UNSCALED_CODECS]
    for enc in encoders:
        enc.reset()
    spikes = torch.stack([
        torch.stack([enc(values[index]) for index, enc in enumerate(encoders)])
        for _ in range(TIMESTEP)
    ])
    return spikes.permute(2, 0, 1)


def build_segments():
    uni = scaled_streams("unipolar")
    bi = scaled_streams("bipolar")
    unscaled = unscaled_streams()
    count = uni.shape[0]
    segments = [(uni[index], bi[index], unscaled[index]) for index in range(count)]
    first = segments[0]
    return [
        tuple(stream[:3] for stream in first),
        tuple(stream[3:] for stream in first),
        *segments[1:],
    ]


def bus(spikes):
    return "".join(str(int(bit)) for bit in reversed(spikes.tolist()))


def main():
    scaled_uni = add_gaines({"polarity": "unipolar", **SCALED})
    scaled_bi = add_gaines({"polarity": "bipolar", **SCALED})
    unscaled = add_gaines({"polarity": "unipolar", **UNSCALED})
    models = (scaled_uni, scaled_bi, unscaled)
    segments = build_segments()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_text(
        "\n".join(
            f"{value:0{SCALED_SELECT_WIDTH}b}" for value in scaled_uni.sel_seq
        )
        + "\n"
    )
    PARAMS.write_text(
        f"`define GEN_SCALED {int(SCALED['scaled'])}\n"
        f"`define GEN_UNSCALED {int(UNSCALED['scaled'])}\n"
        f"`define GEN_SCALED_ENTRY {SCALED_ENTRY}\n"
        f"`define GEN_UNSCALED_ENTRY {UNSCALED_ENTRY}\n"
        f"`define GEN_SCALED_SELECT_WIDTH {SCALED_SELECT_WIDTH}\n"
        f"`define GEN_UNSCALED_SELECT_WIDTH {UNSCALED_SELECT_WIDTH}\n"
        f"`define GEN_PP_DELAY {scaled_uni.hw.pp_delay}\n"
    )

    rows = 0
    resets = 0
    with VEC.open("w") as output:
        for segment in segments:
            for model in models:
                model.reset()
            for cycle, (in_uni, in_bi, in_or) in enumerate(zip(*segment)):
                reset = int(cycle == 0)
                out_uni = int(scaled_uni(in_uni, dim=0).item())
                out_bi = int(scaled_bi(in_bi, dim=0).item())
                out_or = int(unscaled(in_or, dim=0).item())
                output.write(
                    f"{reset} {bus(in_uni)} {out_uni} "
                    f"{bus(in_bi)} {out_bi} {bus(in_or)} {out_or}\n"
                )
                rows += 1
                resets += reset

    print(
        f"wrote {VEC} ({rows} vectors, {resets} resets), "
        f"{PARAMS}, and {ROM} ({len(scaled_uni.sel_seq)} ROM lines)"
    )


if __name__ == "__main__":
    main()
