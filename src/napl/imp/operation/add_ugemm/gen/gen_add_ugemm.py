import math
from pathlib import Path

import torch

from napl.sim.operation import add_ugemm, encode
from napl.syn import translate_node


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_ugemm.vec"
PARAMS = ROOT / "vec" / "add_ugemm_params.vh"
TIMESTEP = 256
ENTRY = 8
COUNT_WIDTH = math.ceil(math.log2(ENTRY + 1))
ACC_WIDTH = math.ceil(math.log2(2 * ENTRY * TIMESTEP + 1)) + 1
# The RTL accumulator holds twice the Python one, so that ACC_WIDTH-bit signed
# register spans the same range as a Python accumulator one bit narrower.
WIDTH = ACC_WIDTH - 1
PROBE_CYCLES = 8
CONFIGS = (
    {"polarity": "unipolar", "scaled": True, "width": WIDTH},
    {"polarity": "bipolar", "scaled": True, "width": WIDTH},
    {"polarity": "unipolar", "scaled": False, "width": WIDTH},
    {"polarity": "bipolar", "scaled": False, "width": WIDTH},
)


# Rail rates driving each input stream to the codec's extremes (all-zeros, the
# rate-0.5 bipolar-zero midpoint, all-ones), fed to every polarity as three
# aligned corner rows that bypass the 1/ENTRY reduction so the non-scaled arm
# saturates too and main() can zip every config by one count.
CORNER_RATES = (0.0, 0.5, 1.0)


def corner_values(polarity):
    return [rate if polarity == "unipolar" else 2 * rate - 1 for rate in CORNER_RATES]


def test_values(config):
    polarity = config["polarity"]
    scaled = config["scaled"]
    low = -0.75 if polarity == "bipolar" else 0.0
    scale = 1.0 if scaled else 1.0 / ENTRY
    known_value = -0.25 if polarity == "bipolar" else 0.25
    known = torch.full((8, ENTRY), known_value * scale)
    fidelity = torch.linspace(low, 0.75, 512).reshape(64, ENTRY) * scale
    corner = torch.tensor([[value] * ENTRY for value in corner_values(polarity)])
    return torch.cat((known, fidelity, corner), dim=0)


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


def check_mapping():
    """Fail if mapping.yaml's width formulas drift from this gen's copies.

    COUNT_WIDTH and ACC_WIDTH live both here and in mapping.yaml. Resolving the
    mapping entry and requiring it to reproduce the gen's values ties the two
    copies together, so editing one formula without the other trips this assert.
    """
    expected = {"SCALED": True, "ENTRY": ENTRY,
                "COUNT_WIDTH": COUNT_WIDTH, "ACC_WIDTH": ACC_WIDTH}
    for rtl_module in ("add_ugemm_unipolar", "add_ugemm_bipolar"):
        binding = translate_node({
            "class": "add_ugemm",
            "rtl_module": rtl_module,
            "config": {"scaled": True, "timestep": TIMESTEP},
            "inputs": {"input": [ENTRY]},
            "dim": -1,
        })
        assert binding.parameters == expected, \
            f"mapping.yaml {rtl_module} resolves {binding.parameters}, not {expected}"


def main():
    check_mapping()
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
