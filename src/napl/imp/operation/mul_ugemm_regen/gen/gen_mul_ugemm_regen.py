"""Generate mul_ugemm_regen vectors and its model-derived direction-vector table.

The RTL produces the RNG levels online with a `sobol` generator, so this script
emits the WIDTH-entry direction-vector table that generator reads rather than the
full 2**WIDTH-point sequence.
"""
import sys
from pathlib import Path

import torch

from napl.sim.operation import mul_ugemm_regen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "mul_ugemm_regen.vec"
PARAMS = ROOT / "vec" / "mul_ugemm_regen_params.vh"
DIRVEC = ROOT / "vec" / "mul_ugemm_regen_dv.hex"

# Configuration mirrors test_mul_ugemm_regen.py.
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
    # p = (x + 1) / 2 over [-1, 1] is the unipolar [0, 1] grid, so the bipolar
    # operands are drawn from a narrower range and the two streams stay distinct.
    value_range = (-1.0, 0.5) if polarity == "bipolar" else None
    return pair_streams(codec_0, codec_1,
                        rep_pairs(polarity, polarity,
                                  range0=value_range, range1=value_range))


def dirvec_rows(num_seq, width, name):
    """Direction vectors of a Sobol sequence, checked against the model's num_seq.

    The hardware generator runs the Antonov-Saleev gray-code recurrence
    x_{n+1} = x_n ^ v[l(n)], where l(n) is the position of the least significant
    zero of the width-bit counter n. The table v is recovered from the model's own
    sequence and replayed here over the whole period, so a sequence the recurrence
    does not reproduce fails the generator instead of the co-simulation.
    """
    period = 2 ** width
    values = num_seq.detach().float().reshape(-1)
    assert values.numel() == period, f'{name} holds {values.numel()} points, not {period}'
    codes = []
    for index in range(period):
        scaled = values[index].item() * period
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f'{name}[{index}] is off the 1/{period} grid'
        codes.append(code)
    assert codes[0] == 0, f'{name} starts at {codes[0]}, not the post-reset 0'

    vectors = [codes[2 ** k] ^ codes[2 ** k - 1] for k in range(width)]
    state = 0
    for index in range(period):
        assert state == codes[index], \
            f'{name} is not a gray-code Sobol sequence: the recurrence gives {state} ' \
            f'at index {index}, the model gives {codes[index]}'
        # The all-ones counter state takes the top position, which returns to 0.
        position = width - 1 if index == period - 1 else (~index & (index + 1)).bit_length() - 1
        state ^= vectors[position]
    assert state == 0, f'{name} returns to {state} on the wrap, not 0'
    return [f'{vector:0{width}b}' for vector in vectors]


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
    assert streams["unipolar"] != streams["bipolar"], \
        "bipolar stimulus is identical to unipolar"

    models = {
        polarity: mul_ugemm_regen(dict(OP_CONFIG, polarity=polarity))
        for polarity in ("unipolar", "bipolar")
    }
    for model in models.values():
        model.reset()

    # rng_seq is the encoder's num_seq scaled by the depth; the recurrence check
    # works on the unscaled 1/2**WIDTH grid. One table serves every sobol
    # instance, so require the polarities to share the sequence.
    assert torch.equal(models["unipolar"].rng_seq, models["bipolar"].rng_seq), \
        "the two polarities no longer share one rng_seq"
    dirvec = dirvec_rows(models["unipolar"].rng_seq.div(2**WIDTH), WIDTH,
                         "mul_ugemm_regen rng_seq")
    DIRVEC.parent.mkdir(parents=True, exist_ok=True)
    DIRVEC.write_text("\n".join(dirvec) + "\n")
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

    PARAMS.write_text(
        f"`define GEN_WIDTH {WIDTH}\n"
        f"`define GEN_PP_DELAY {models['unipolar'].hw.pp_delay}\n"
        f"`define GEN_VECTORS {rows}\n"
    )

    print(
        f"wrote {VEC} ({rows} vectors, reset@0/{reset_at}), "
        f"{PARAMS} and {DIRVEC} ({len(dirvec)} direction vectors, WIDTH={WIDTH})"
    )


if __name__ == "__main__":
    main()
