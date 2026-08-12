"""
Generate golden test vectors for the conv_gaines module RTL straight from napl's
functional Python model (napl.sim.module.conv_gaines) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

conv_gaines composes linear_gaines over the im2col patch: one lane per output
pixel-channel (batch, out_channel, out_row, out_col), whose addends are the
Gaines products of the patch spikes with the layer's own weight streams plus the
optional bias spike, reduced by the Gaines adder. The RTL therefore wires the
patch and instantiates one linear_gaines_<polarity> per spatial output position,
with OUT_CHANNELS lanes each -- a module-layer circuit, instantiated rather than
rebuilt.

Shared indices: the model holds one core, so one threshold index, one bias
encoder index and one Gaines select index serve every output position. Each RTL
position holds its own copy of all three, and each copy advances by one
unconditionally on every timestep, so the copies stay equal to the model's and
the outputs are bit-exact with the shared-core model.

Weight and bias are held fixed-point codes rather than spike streams (the model
sets internal_encode = True: it generates both streams itself), so they go to
../vec/conv_gaines_operand.hex instead of vector columns. One file serves both
polarity DUTs: an operand code is the probability code either polarity compares
against.

Bipolar zero padding is the model's decorrelated rate-0.5 stream, not a
deterministic toggle: sim/module/conv_gaines.py:150-159 builds a separate
pad_encoder on sequence dimension dim + K + 2 and thresholds it at 0.5, and
line 213 reads it at (timestep_cur - 1) % pad_len. The RTL pad cell is that
encoder -- an `encode` cell comparing the constant 0.5 against ../vec/pad_rom.hex
-- and this file writes that ROM from the model's own pad_encoder.num_seq and
asserts it differs from every weight and bias sequence, which is the
decorrelation the model relies on. Unipolar padding is a zero spike, because the
unipolar path pads the input tensor with 0.

The composed linear_gaines cells $readmemb their tables from paths relative to
the simulation cwd (module/conv_gaines/), so this file also writes
vec/linear_gaines_w.hex (the per-tap weight thresholds), vec/encode_rom.hex (the
bias sequence) and vec/gaines_rom.hex (the Gaines select sequence) from the built
model.

Six DUT configurations share each row, one column of expected output each. Every
sequence dimension assignment above is fixed by the fan-in K, so the geometries
below keep K constant and vary padding, stride, dilation, bias, and the adder:

  u_a -- unipolar, padding 1, stride 1, dilation 1, bias, scaled   (entry 4)
  u_b -- unipolar, padding 0, stride 1, dilation 1, no bias, OR    (entry 3)
  u_c -- unipolar, padding 1, stride 2, dilation 2, bias, scaled   (entry 4)
  b_a -- bipolar,  padding 1, stride 1, dilation 1, bias, scaled   (pad stream)
  b_b -- bipolar,  padding 0, stride 1, dilation 1, bias, scaled   (no pad)
  b_c -- bipolar,  padding 1, stride 2, dilation 2, bias, scaled   (pad stream)

Non-scaled bipolar addition is rejected by the model, so the bipolar arms are
scaled throughout. The unit carries no headroom parameter: the Gaines adder has
no accumulator, so every elaboration parameter is observable through a golden
column width, the lane guard, or a changed output, and no saturation block is
needed.

Polarity selects the circuit (separate `_unipolar` / `_bipolar` modules), so the
two polarities run as parallel columns: each row drives both polarity DUTs with
their own encoded input stream. Three sequences follow each other: one over the
first input tensor, one over the second, then the first replayed after the second
has dirtied the state and reset() has cleared it, which must reproduce the first
sequence bit for bit. The dirty block is an odd number of timesteps, so the pad,
bias and select indices are all left off their reset values.

Output: ../vec/conv_gaines.vec, one line per timestep:

    <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c> <out_b_a> <out_b_b> <out_b_c>

`in_*` is BATCH*IN_CHANNELS*IN_H*IN_W binary digits, MSB first, so input element
(b, ic, ih, iw) occupies i_input[((b*IN_CHANNELS + ic)*IN_H + ih)*IN_W + iw].
Each `out_*` column is that configuration's LANES digits with lane (b, oc, oh, ow)
row-major.

Sizing values come from this file only and are emitted into
../vec/conv_gaines_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay and the
row count, which the testbench compares against the rows it consumed.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_conv_gaines.py
"""
import math
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import conv_gaines
from napl.sim.operation import encode
from napl.syn import translate_node

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "conv_gaines.vec"
PARAMS = VEC_DIR / "conv_gaines_params.vh"

# Shapes and codec mirror tests/module/test_conv_gaines.py's fidelity-scale input at small channel and position counts, with a 3x1 kernel so K = 3 keeps the scaled entry K + 1 a power of two.
SHAPE = (1, 1, 5, 4)          # batch, in_channels, height, width
OUT_CHANNELS = 2
KERNEL = (3, 1)
TIMESTEP = 256

BATCH, IN_CHANNELS, IN_H, IN_W = SHAPE
K = IN_CHANNELS * KERNEL[0] * KERNEL[1]
IN_WIDTH = BATCH * IN_CHANNELS * IN_H * IN_W

SEQ_WIDTH = math.ceil(math.log2(TIMESTEP))
LEN = 2 ** SEQ_WIDTH
# Every sequence index wraps at LEN, so a block is one full period.
TIMESTEPS = LEN
# An odd dirty block leaves the pad, bias and select indices off their reset
# values, so a reset that misses any of them shows up in the replayed block.
DIRTY = TIMESTEPS // 2 + 1
# round(log2(entry)) of the scaled arms, the model's own scale width.
SCALE_WIDTH = round(math.log2(K + 1))

# label, padding, stride, dilation, has_bias, scaled
ARM_SPECS = {
    "unipolar": (("a", 1, 1, 1, True, True),
                 ("b", 0, 1, 1, False, False),
                 ("c", 1, 2, 2, True, True)),
    "bipolar": (("a", 1, 1, 1, True, True),
                ("b", 0, 1, 1, True, True),
                ("c", 1, 2, 2, True, True)),
}


def out_hw(padding, stride, dilation):
    """Output height and width of one geometry, as the RTL derives them."""
    height = (IN_H + 2 * padding - dilation * (KERNEL[0] - 1) - 1) // stride + 1
    width = (IN_W + 2 * padding - dilation * (KERNEL[1] - 1) - 1) // stride + 1
    return height, width


def lanes_of(padding, stride, dilation):
    """Output pixel-channels one geometry elaborates."""
    height, width = out_hw(padding, stride, dilation)
    return BATCH * OUT_CHANNELS * height * width


def operand_codes(count, stride, offset):
    """Return fixed-point codes in ``[0, LEN]`` on the probability grid."""
    return [(index * stride + offset) % (LEN + 1) for index in range(count)]


WEIGHT_CODES = operand_codes(OUT_CHANNELS * K, 29, 5)
BIAS_CODES = operand_codes(OUT_CHANNELS, 53, 7)
INPUT_CODES = {
    "unipolar": [operand_codes(IN_WIDTH, 71, 13), operand_codes(IN_WIDTH, 37, 101)],
    "bipolar": [operand_codes(IN_WIDTH, 43, 97), operand_codes(IN_WIDTH, 59, 151)],
}


def to_value(code, polarity):
    """Map a fixed-point probability code to the polarity's real value."""
    probability = code / LEN
    return probability if polarity == "unipolar" else 2.0 * probability - 1.0


def tensor(codes, shape, polarity):
    """Build a model tensor from operand codes, in the polarity's value domain."""
    values = [to_value(code, polarity) for code in codes]
    return torch.tensor(values, dtype=global_config.ntype).reshape(shape)


def as_binary(bits):
    """Render a bit list as a Verilog %b string, highest bit index first."""
    return "".join(str(int(bit)) for bit in reversed(bits))


def bits_of(spike):
    """Flatten a spike tensor to a Python bit list in lane order."""
    return spike.reshape(-1).to(torch.int64).tolist()


def codes_of(sequence, name):
    """Render a model number sequence as integer codes on the 1/LEN grid."""
    codes = []
    for index, value in enumerate(sequence.detach().float().reshape(-1).tolist()):
        scaled = value * LEN
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f"{name}[{index}] is off the 1/{LEN} grid"
        codes.append(code)
    return codes


class Capture:
    """Forward hook that keeps the most recent output of an encoder."""


    def __init__(self, module):
        self.spike = None
        module.register_forward_hook(self)


    def __call__(self, module, args, output):
        self.spike = output


class PolarityArm:
    """One polarity's input encoder plus its three conv_gaines geometries."""


    def __init__(self, polarity):
        self.polarity = polarity
        self.specs = ARM_SPECS[polarity]
        codec = {"polarity": polarity, "timestep": TIMESTEP,
                 "generator": "sobol", "dim": 1}
        self.weight = tensor(WEIGHT_CODES, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity)
        self.bias = tensor(BIAS_CODES, (OUT_CHANNELS,), polarity)
        self.values = [tensor(codes, SHAPE, polarity) for codes in INPUT_CODES[polarity]]
        self.encoder = encode(codec)
        # The input spikes the layers reduce are captured from the encoder's own
        # forward, so the golden column is provably the bits the model saw.
        self.input_spike = Capture(self.encoder)
        self.configs = [{"polarity": polarity, "timestep": TIMESTEP,
                         "generator": "sobol", "dim": 2, "scaled": scaled}
                        for *_, scaled in self.specs]
        self.layers = [
            conv_gaines(self.weight, self.bias if has_bias else None,
                        stride=stride, padding=padding, dilation=dilation,
                        config=config)
            for (_, padding, stride, dilation, has_bias, _), config
            in zip(self.specs, self.configs)]
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Every elaborated geometry is translated: they differ in
        padding, stride, dilation, bias, and the adder, which change LANES,
        HAS_BIAS and SCALED.
        """
        for (_, padding, stride, dilation, has_bias, scaled), config, layer in zip(
                self.specs, self.configs, self.layers):
            node = {"class": "conv_gaines",
                    "config": {"weight": self.weight,
                               "bias": self.bias if has_bias else None,
                               "stride": stride, "padding": padding,
                               "dilation": dilation, "config": config,
                               "lanes": lanes_of(padding, stride, dilation)},
                    "inputs": {"input": {"shape": SHAPE}}}
            expected = {"BATCH": BATCH, "IN_CHANNELS": IN_CHANNELS,
                        "IN_H": IN_H, "IN_W": IN_W,
                        "OUT_CHANNELS": OUT_CHANNELS,
                        "KERNEL_H": KERNEL[0], "KERNEL_W": KERNEL[1],
                        "STRIDE": stride, "PADDING": padding, "DILATION": dilation,
                        "SEQ_WIDTH": SEQ_WIDTH, "HAS_BIAS": int(has_bias),
                        "SCALE_WIDTH": SCALE_WIDTH,
                        "LANES": lanes_of(padding, stride, dilation)}
            if self.polarity == "unipolar":
                expected["SCALED"] = int(scaled)
            binding = translate_node(node)
            assert binding.rtl_module == f"conv_gaines_{self.polarity}", binding.rtl_module
            assert binding.parameters == expected, \
                f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"
            assert layer.entry == K + int(has_bias), layer.entry


    def reset(self):
        """Restart the input encoder and every layer."""
        self.encoder.reset()
        for layer in self.layers:
            layer.reset()


    def step(self, index):
        """Advance one timestep over one input and return the row columns."""
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
    for timestep in range(TIMESTEPS):
        emit_row(rows, arms, index, timestep == 0)
    return arms


def write_roms(arms):
    """Emit the weight, bias, select and pad tables from model-owned state."""
    reference = arms[0].layers[0].core
    for arm in arms:
        for layer, (label, _, _, _, has_bias, scaled) in zip(arm.layers, arm.specs):
            assert torch.equal(layer.core.w_num_seq, reference.w_num_seq), \
                f"the {arm.polarity} {label} arm holds a different weight table"
            if has_bias:
                assert torch.equal(layer.core.b_encoder.num_seq,
                                   reference.b_encoder.num_seq), \
                    f"the {arm.polarity} {label} arm holds a different bias sequence"
            if scaled:
                assert layer.core.acc.sel_seq == reference.acc.sel_seq, \
                    f"the {arm.polarity} {label} arm holds a different select sequence"

    thresholds = reference.w_num_seq.detach().float()
    assert tuple(thresholds.shape) == (LEN, K), thresholds.shape
    weight_rows = []
    for timestep in range(LEN):
        codes = codes_of(thresholds[timestep], f"weight sequence [{timestep}]")
        weight_rows.append("".join(f"{code:0{SEQ_WIDTH}b}" for code in reversed(codes)))
    (VEC_DIR / "linear_gaines_w.hex").write_text(
        "\n".join(weight_rows) + "\n", encoding="utf-8"
    )

    bias_codes = codes_of(reference.b_encoder.num_seq, "bias sequence")
    (VEC_DIR / "encode_rom.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH}b}" for code in bias_codes) + "\n",
        encoding="utf-8",
    )
    (VEC_DIR / "gaines_rom.hex").write_text(
        "\n".join(f"{index:0{SCALE_WIDTH}b}" for index in reference.acc.sel_seq) + "\n",
        encoding="utf-8",
    )

    # The pad stream lives on its own sequence dimension to keep it from correlating with the weight and bias streams, and its table is asserted below to differ from every one of them.
    pad_layer = next(layer for arm in arms if arm.polarity == "bipolar"
                     for layer in arm.layers if hasattr(layer, "pad_encoder"))
    assert pad_layer.pad_len == LEN, pad_layer.pad_len
    pad_codes = codes_of(pad_layer.pad_encoder.num_seq, "pad sequence")
    assert pad_codes != bias_codes, "the pad stream shares the bias sequence"
    for feature in range(K):
        assert pad_codes != codes_of(thresholds[:, feature], f"weight column {feature}"), \
            f"the pad stream shares the weight sequence of tap {feature}"
    (VEC_DIR / "pad_rom.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH}b}" for code in pad_codes) + "\n",
        encoding="utf-8",
    )
    # The RTL pad cell compares the constant 0.5 against that ROM, which is the
    # model's own pad bit at every timestep.
    half = LEN // 2
    expected = [float(code < half) for code in pad_codes]
    assert pad_layer.pad_bits == expected, "the pad ROM does not reproduce pad_bits"


def write_operands():
    """Emit held weight codes followed by held bias codes."""
    codes = list(WEIGHT_CODES) + list(BIAS_CODES)
    (VEC_DIR / "conv_gaines_operand.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in codes) + "\n",
        encoding="utf-8",
    )


def main():
    """Write vectors, ROMs, and the testbench parameter header."""
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["rst", "in_u", "in_b",
              "out_u_a", "out_u_b", "out_u_c", "out_b_a", "out_b_b", "out_b_c"]
    rows = [" ".join(header)]
    arms = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=DIRTY)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:1 + 3 * TIMESTEPS], \
        "reset did not reproduce the opening sequence"
    columns = [row.split() for row in rows[1:]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        "bipolar input stimulus is identical to unipolar"

    write_roms(arms)
    write_operands()
    pp_delay = arms[0].layers[0].hw.pp_delay
    defines = [
        ("BATCH", BATCH),
        ("IN_CHANNELS", IN_CHANNELS),
        ("IN_H", IN_H),
        ("IN_W", IN_W),
        ("OUT_CHANNELS", OUT_CHANNELS),
        ("KERNEL_H", KERNEL[0]),
        ("KERNEL_W", KERNEL[1]),
        ("SEQ_WIDTH", SEQ_WIDTH),
        ("SCALE_WIDTH", SCALE_WIDTH),
        ("PP_DELAY", pp_delay),
        ("VECTORS", len(rows) - 1),
    ]
    for polarity, prefix in (("unipolar", "U"), ("bipolar", "B")):
        for label, padding, stride, dilation, has_bias, scaled in ARM_SPECS[polarity]:
            suffix = f"{prefix}_{label.upper()}"
            defines.extend((
                (f"PADDING_{suffix}", padding),
                (f"STRIDE_{suffix}", stride),
                (f"DILATION_{suffix}", dilation),
                (f"HAS_BIAS_{suffix}", int(has_bias)),
                (f"LANES_{suffix}", lanes_of(padding, stride, dilation)),
            ))
            if polarity == "unipolar":
                defines.append((f"SCALED_{suffix}", int(scaled)))
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
