"""Generate linear_gaines module vectors from the Python simulation model.

The co-simulation covers the three legal unipolar configurations needed to
exercise scaled MUX addition, non-scaled OR addition, and the optional bias. The
bipolar RTL is intrinsically scaled, matching the Python class, so it has one
scaled-with-bias arm. Each recorded block runs from reset for a full sequence;
the opening block is replayed after a different input has dirtied every encoder
index.
"""
import math
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import linear_gaines
from napl.sim.operation import encode
from napl.syn import translate_node

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "linear_gaines.vec"
PARAMS = VEC_DIR / "linear_gaines_params.vh"

# The scaled arm includes a bias, making its entry 16 as required by add_gaines.
IN_FEATURES = 15
LANES = 8
TIMESTEP = 256
SEQ_WIDTH = math.ceil(math.log2(TIMESTEP))
W_LEN = 2 ** SEQ_WIDTH
SCALE_WIDTH = round(math.log2(IN_FEATURES + 1))
DIRTY = W_LEN // 2 + 1

# label, scaled, has_bias
ARM_SPECS = {
    "unipolar": (("a", True, True), ("b", False, False), ("c", False, True)),
    "bipolar": (("a", True, True),),
}


def operand_codes(count, stride, offset):
    """Return fixed-point codes in ``[0, W_LEN]`` on the probability grid."""
    return [(index * stride + offset) % (W_LEN + 1) for index in range(count)]


WEIGHT_CODES = operand_codes(LANES * IN_FEATURES, 29, 5)
BIAS_CODES = operand_codes(LANES, 53, 7)
INPUT_CODES = {
    "unipolar": [operand_codes(IN_FEATURES, 71, 13),
                  operand_codes(IN_FEATURES, 37, 101)],
    "bipolar": [operand_codes(IN_FEATURES, 43, 97),
                 operand_codes(IN_FEATURES, 59, 151)],
}


def to_value(code, polarity):
    """Map a probability code to the polarity's numeric domain."""
    probability = code / W_LEN
    return probability if polarity == "unipolar" else 2.0 * probability - 1.0


def tensor(codes, shape, polarity):
    """Build a model tensor from fixed-point probability codes."""
    values = [to_value(code, polarity) for code in codes]
    return torch.tensor(values, dtype=global_config.ntype).reshape(shape)


def as_binary(bits):
    """Render a bit list as a Verilog binary string, highest index first."""
    return "".join(str(int(bit)) for bit in reversed(bits))


def bits_of(spike):
    """Flatten a spike tensor to a Python bit list in lane order."""
    return spike.reshape(-1).to(torch.int64).tolist()


class Capture:
    """Keep the most recent output of a hooked encoder."""

    def __init__(self, module):
        self.spike = None
        module.register_forward_hook(self)

    def __call__(self, module, args, output):
        self.spike = output


class PolarityArm:
    """Hold one input encoder and every legal arm for one polarity."""

    def __init__(self, polarity):
        self.polarity = polarity
        self.specs = ARM_SPECS[polarity]
        self.weight = tensor(WEIGHT_CODES, (LANES, IN_FEATURES), polarity)
        self.bias = tensor(BIAS_CODES, (LANES,), polarity)
        self.values = [tensor(codes, (IN_FEATURES,), polarity)
                       for codes in INPUT_CODES[polarity]]
        self.encoder = encode({"polarity": polarity, "timestep": TIMESTEP,
                               "generator": "sobol", "dim": 1})
        self.input_spike = Capture(self.encoder)
        self.configs = []
        self.layers = []
        for _, scaled, has_bias in self.specs:
            config = {"polarity": polarity, "timestep": TIMESTEP,
                      "generator": "sobol", "dim": 2, "scaled": scaled}
            self.configs.append(config)
            self.layers.append(linear_gaines(
                self.weight, self.bias if has_bias else None, config
            ))
        self.reset()

    def check_mapping(self):
        """Require the mapping to resolve the parameters used by each arm."""
        for (_, scaled, has_bias), config in zip(self.specs, self.configs):
            expected = {
                "IN_FEATURES": IN_FEATURES,
                "LANES": LANES,
                "SEQ_WIDTH": SEQ_WIDTH,
                "HAS_BIAS": int(has_bias),
                "SCALE_WIDTH": SCALE_WIDTH,
            }
            if self.polarity == "unipolar":
                expected["SCALED"] = int(scaled)
            node = {
                "class": "linear_gaines",
                "config": {
                    "weight": self.weight,
                    "bias": self.bias if has_bias else None,
                    "config": config,
                    "lanes": LANES,
                },
            }
            binding = translate_node(node)
            assert binding.rtl_module == f"linear_gaines_{self.polarity}", \
                binding.rtl_module
            assert binding.parameters == expected, \
                f"mapping.yaml linear_gaines resolves {binding.parameters}"

    def reset(self):
        """Restart the input encoder and every layer instance."""
        self.encoder.reset()
        for layer in self.layers:
            layer.reset()

    def step(self, index):
        """Advance one timestep over one input and return the vector columns."""
        spike = self.encoder(self.values[index])
        outputs = [bits_of(layer(spike)) for layer in self.layers]
        return bits_of(self.input_spike.spike), outputs


def emit_row(rows, arms, index, first):
    """Step both polarities and append one golden row."""
    inputs = []
    outputs = []
    for arm in arms:
        input_bits, output_bits = arm.step(index)
        inputs.append(as_binary(input_bits))
        outputs.extend(as_binary(bits) for bits in output_bits)
    rows.append(" ".join(["1" if first else "0"] + inputs + outputs))


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence for both polarities."""
    arms = [PolarityArm("unipolar"), PolarityArm("bipolar")]
    for arm in arms:
        arm.check_mapping()
    if dirty:
        for _ in range(dirty):
            for arm in arms:
                arm.step(1 - index)
        for arm in arms:
            arm.reset()
    for timestep in range(W_LEN):
        emit_row(rows, arms, index, timestep == 0)
    return arms


def sequence_rows(sequence, width, name):
    """Render a model number sequence as fixed-width binary ROM rows."""
    values = sequence.detach().float().reshape(-1)
    count = values.numel()
    rows = []
    for index, value in enumerate(values):
        scaled = value.item() * count
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, \
            f"{name}[{index}] is off the 1/{count} grid"
        rows.append(f"{code:0{width}b}")
    return rows


def write_roms(arms):
    """Emit weight, bias, and Gaines selection ROMs from model-owned state."""
    reference = arms[0].layers[0]
    for arm in arms:
        for layer, (label, scaled, has_bias) in zip(arm.layers, arm.specs):
            assert torch.equal(layer.w_num_seq,
                               reference.w_num_seq), \
                f"the {arm.polarity} {label} arm holds a different weight table"
            if has_bias:
                assert torch.equal(layer.b_encoder.num_seq,
                                   reference.b_encoder.num_seq), \
                    f"the {arm.polarity} {label} arm holds a different bias sequence"
            if scaled:
                assert layer.acc.sel_seq == reference.acc.sel_seq, \
                    f"the {arm.polarity} {label} arm holds a different select sequence"

    thresholds = reference.w_num_seq.detach().float()
    assert tuple(thresholds.shape) == (W_LEN, IN_FEATURES), thresholds.shape
    weight_rows = []
    for timestep in range(W_LEN):
        codes = []
        for feature in reversed(range(IN_FEATURES)):
            scaled = thresholds[timestep, feature].item() * W_LEN
            code = round(scaled)
            assert abs(scaled - code) < 1e-9, \
                f"weight sequence [{timestep}, {feature}] is off the grid"
            codes.append(f"{code:0{SEQ_WIDTH}b}")
        weight_rows.append("".join(codes))
    (VEC_DIR / "linear_gaines_w.hex").write_text(
        "\n".join(weight_rows) + "\n", encoding="utf-8"
    )
    (VEC_DIR / "encode_rom.hex").write_text(
        "\n".join(sequence_rows(reference.b_encoder.num_seq,
                                SEQ_WIDTH, "bias sequence")) + "\n",
        encoding="utf-8",
    )
    (VEC_DIR / "gaines_rom.hex").write_text(
        "\n".join(f"{index:0{SCALE_WIDTH}b}" for index in reference.acc.sel_seq) + "\n",
        encoding="utf-8",
    )


def write_operands():
    """Emit held weight codes followed by held bias codes."""
    codes = list(WEIGHT_CODES) + list(BIAS_CODES)
    (VEC_DIR / "linear_gaines_operand.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in codes) + "\n",
        encoding="utf-8",
    )


def main():
    """Write vectors, ROMs, and the testbench parameter header."""
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["rst", "in_u", "in_b", "out_u_a", "out_u_b", "out_u_c", "out_b_a"]
    rows = [" ".join(header)]
    arms = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=DIRTY)
    assert rows[1:1 + W_LEN] == rows[1 + 2 * W_LEN:1 + 3 * W_LEN], \
        "reset did not reproduce the opening sequence"
    columns = [row.split() for row in rows[1:]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        "bipolar input stimulus is identical to unipolar"

    write_roms(arms)
    write_operands()
    pp_delay = arms[0].layers[0].hw.pp_delay
    defines = [
        ("IN_FEATURES", IN_FEATURES),
        ("LANES", LANES),
        ("SEQ_WIDTH", SEQ_WIDTH),
        ("SCALE_WIDTH", SCALE_WIDTH),
        ("PP_DELAY", pp_delay),
        ("VECTORS", len(rows) - 1),
    ]
    for label, scaled, has_bias in ARM_SPECS["unipolar"]:
        suffix = label.upper()
        defines.extend(((f"SCALED_{suffix}", int(scaled)),
                        (f"HAS_BIAS_{suffix}", int(has_bias))))
    defines.append(("HAS_BIAS_BIPOLAR", 1))
    PARAMS.write_text(
        "".join(f"`define GEN_{name} {value}\n" for name, value in defines),
        encoding="utf-8",
    )
    VEC.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} ("
        + ", ".join(f"GEN_{name}={value}" for name, value in defines)
        + ")"
    )


if __name__ == "__main__":
    main()
