import math
from pathlib import Path

import torch

from napl.sim.operation import encode
from napl.sim.operation import add_ugemm


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_ugemm.vec"
PARAMS = ROOT / "vec" / "add_ugemm_params.vh"
TIMESTEP = 256
ENTRY = 8
COUNT_WIDTH = math.ceil(math.log2(ENTRY + 1))
ACC_WIDTH = math.ceil(math.log2(2 * ENTRY * TIMESTEP + 1)) + 1
PROBE_CYCLES = 8
CONFIGS = (
    {"polarity": "unipolar", "scaled": True},
    {"polarity": "bipolar", "scaled": True},
    {"polarity": "unipolar", "scaled": False},
    {"polarity": "bipolar", "scaled": False},
)


def test_values(config):
    polarity = config["polarity"]
    scaled = config["scaled"]
    low = -0.75 if polarity == "bipolar" else 0.0
    scale = 1.0 if scaled else 1.0 / ENTRY
    known_value = -0.25 if polarity == "bipolar" else 0.25
    known = torch.full((8, ENTRY), known_value * scale)
    fidelity = torch.linspace(low, 0.75, 512).reshape(64, ENTRY) * scale
    return torch.cat((known, fidelity), dim=0)


def encode_segments(config):
    codec = {
        "polarity": config["polarity"],
        "timestep": TIMESTEP,
        "generator": "sobol",
        "dim": 1,
    }
    segments = []
    for values_row in test_values(config):
        enc = encode(dict(codec))
        enc.reset()
        segments.append([enc(values_row).clone() for _ in range(TIMESTEP)])
    return segments


def bus(spikes):
    return "".join(str(int(bit)) for bit in reversed(spikes.tolist()))


def main():
    models = tuple(add_ugemm(dict(config)) for config in CONFIGS)
    segments = tuple(encode_segments(config) for config in CONFIGS)
    segment_count = len(segments[0])

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_SCALED {int(CONFIGS[0]['scaled'])}\n"
        f"`define GEN_UNSCALED {int(CONFIGS[2]['scaled'])}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_COUNT_WIDTH {COUNT_WIDTH}\n"
        f"`define GEN_ACC_WIDTH {ACC_WIDTH}\n"
        f"`define GEN_PP_DELAY {models[0].hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for model in models:
            model.reset()
        for cycle in range(PROBE_CYCLES):
            spikes = tuple(variant[8][cycle] for variant in segments)
            results = []
            for model, input_spikes in zip(models, spikes):
                results.append(int(model(input_spikes, dim=-1).item()))
            output.write(
                f"{int(cycle == 0)} "
                + " ".join(
                    f"{bus(input_spikes)} {result}"
                    for input_spikes, result in zip(spikes, results)
                )
                + "\n"
            )

        assert all(model.accumulator.ne(0).any().item() for model in models)

        for segment_index in range(segment_count):
            for model in models:
                model.reset()
            for cycle in range(TIMESTEP):
                spikes = tuple(
                    variant[segment_index][cycle] for variant in segments
                )
                results = []
                for model, input_spikes in zip(models, spikes):
                    results.append(int(model(input_spikes, dim=-1).item()))

                output.write(
                    f"{int(cycle == 0)} "
                    + " ".join(
                        f"{bus(input_spikes)} {result}"
                        for input_spikes, result in zip(spikes, results)
                    )
                    + "\n"
                )

    vector_count = PROBE_CYCLES + segment_count * TIMESTEP
    print(
        f"wrote {VEC} ({vector_count} vectors, "
        f"{segment_count + 1} reset segments) and {PARAMS} "
        f"(ENTRY={ENTRY}, ACC_WIDTH={ACC_WIDTH})"
    )


if __name__ == "__main__":
    main()
