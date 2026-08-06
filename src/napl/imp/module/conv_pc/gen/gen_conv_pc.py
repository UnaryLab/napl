"""
Generate golden test vectors for the conv_pc module RTL straight from napl's
functional Python model (napl.sim.module.conv_pc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

conv_pc reduces one lane per output pixel-channel (batch, out_channel, out_row,
out_col) and publishes the lane's parallel count instead of a spike. The count is
the same popcount conv builds and hands to add_any:

    unipolar   count = sum_k x_k * w_k                      = popcount(AND(x, w))
    bipolar    count = sum_k x_k*w_k + sum_k (1-x_k)(1-w_k)
                     = sum_k (1 - x_k - w_k + 2*x_k*w_k)    = popcount(XNOR(x, w))

with the bias spike added as one more addend. So conv_pc is conv minus the
add_any stage, and the RTL lane is K = in_channels*kernel_h*kernel_w
mul_gaines_<polarity> cells plus the optional bias bit feeding a popcount:

    mul_gaines_<polarity>  -- one per (output position, patch tap)
    popcount               -- one per output position, ENTRY addends in, COUNT_W bits out

The popcount is written in the module because the operation layer holds no
standalone popcount circuit: add_any bundles it with the scaled accumulator
conv_pc does not have.

The im2col patch is wiring in the RTL, so the input port carries the NCHW spike
tensor itself. A tap that falls outside the input reads the pad port on the
bipolar variant, whose bit the model draws from a separate rate-0.5 encoder on
its own sobol dimension (a deterministic toggle would correlate with the weight
stream). The unipolar variant has no pad port: its zero pad is a constant zero
spike, so only `pad_b` is a column.

The model re-encodes the weight, the bias, and the pad stream from externally
held tensors every timestep, so all three arrive on RTL ports rather than being
rebuilt in hardware. The weight and bias spikes are captured from the model's own
encoders with forward hooks, so the columns are the exact bits the model reduced;
the pad bit is read from the model's own precomputed pad sequence at the index
the model used this timestep (the pad encoder is not called in forward, so a hook
cannot see it).

Six DUT configurations share each row, one column of expected output each: three
geometries per polarity, chosen so padding, bias, stride, and dilation each vary.

  a  -- padding 1, stride 1, dilation 1, with bias  (pad stream, entry = K + 1)
  b  -- padding 0, stride 1, dilation 1, no bias    (no padded taps, entry = K)
  c  -- padding 2, stride 2, dilation 2, with bias  (strided, dilated, padded)

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded streams. Three sequences follow
each other: one over the first input tensor, one over the second, then the first
replayed after the second has advanced the encoders and reset() has restarted
them, which must reproduce the first sequence bit for bit. conv_pc holds no
accumulator, so the RTL is stateless and carries no reset port: that replay checks
the Python encoders the vectors are drawn from, not RTL reset behavior.

Output: ../vec/conv_pc.vec, one line per timestep:

    <in_u> <in_b> <w_u> <w_b> <bias_u> <bias_b> <pad_b>
    <out_u_a> <out_u_b> <out_u_c> <out_b_a> <out_b_b> <out_b_c>

`in_*` is BATCH*IN_CHANNELS*IN_H*IN_W binary digits, MSB first, so input element
(b, ic, ih, iw) occupies i_input_spike[((b*IN_CHANNELS + ic)*IN_H + ih)*IN_W + iw].
`w_*` is OUT_CHANNELS*K digits with out channel oc, tap t at i_weight[oc*K + t],
the tap order being (in_channel, kernel_row, kernel_col) row-major. `bias_*` is
OUT_CHANNELS digits and `pad_b` one digit.

Each `out_*` column is a *count bus*, not one bit per lane: lane (b, oc, oh, ow)
in row-major order holds a COUNT_W-bit unsigned count at
o_out[lane*COUNT_W +: COUNT_W], so the column is LANES*COUNT_W digits, highest bit
index first like every other column. The testbench checks that character count against LANES*COUNT_W. That check is an
echo of COUNT_W rather than a check on it, since count_width() below sizes both
the column and the elaborated parameter; COUNT_W is pinned by the RTL's
elaboration guard against clog2(ENTRY + 1) and by check_mapping() against the
mapping.yaml expression.

Sizing values come from this file only and are emitted into
../vec/conv_pc_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay and the
row count, which the testbench compares against the rows it consumed.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_conv_pc.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import conv_pc
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "conv_pc.vec"
PARAMS = VEC_DIR / "conv_pc_params.vh"

# Shapes and codec mirror tests/module/test_conv_pc.py's fidelity-scale checks,
# scaled down to keep the elaborated lane count small.
SHAPE = (1, 2, 6, 6)          # batch, in_channels, height, width
OUT_CHANNELS = 3
KERNEL = (3, 3)
TIMESTEP = 256
TIMESTEPS = TIMESTEP

BATCH, IN_CHANNELS, IN_H, IN_W = SHAPE
K = IN_CHANNELS * KERNEL[0] * KERNEL[1]

# label, padding, stride, dilation, has_bias
GEOMETRY = [
    ("a", 1, 1, 1, True),
    ("b", 0, 1, 1, False),
    ("c", 2, 2, 2, True),
]


def clog2(value):
    """Bits needed to hold 0..value-1, matching the RTL clog2 function."""
    width = 0
    remaining = value - 1
    while remaining > 0:
        width += 1
        remaining >>= 1
    return width


def count_width(has_bias):
    """Bits per lane count: clog2(entry + 1) for entry = K + has_bias."""
    return clog2(K + int(has_bias) + 1)


# Neither entry here is a power of two, so clog2(entry) and clog2(entry + 1) give
# the same width and these vectors cannot tell the + 1 apart. linear_pc's no-bias
# arm carries the power-of-two entry that pins it.


def out_hw(padding, stride, dilation):
    """Output height and width of one geometry, as the RTL derives them."""
    height = (IN_H + 2 * padding - dilation * (KERNEL[0] - 1) - 1) // stride + 1
    width = (IN_W + 2 * padding - dilation * (KERNEL[1] - 1) - 1) // stride + 1
    return height, width


def lanes_of(padding, stride, dilation):
    """Output positions one geometry elaborates."""
    height, width = out_hw(padding, stride, dilation)
    return BATCH * OUT_CHANNELS * height * width


# Bipolar values run over a narrower range than unipolar ones, so the encoded
# bipolar rate (x + 1) / 2 differs from the unipolar rate x.
RANGE = {'unipolar': (0.0, 1.0), 'bipolar': (-1.0, 0.75)}


def grid(count, shape, polarity, offset):
    """A deterministic operand tensor spanning the polarity's value range."""
    low, high = RANGE[polarity]
    steps = torch.linspace(low, high, count, dtype=global_config.ntype)
    return steps.roll(offset).reshape(shape)


def as_binary(bits):
    """Render a bit list as a Verilog %b string, highest bit index first."""
    return "".join(str(int(bit)) for bit in reversed(bits))


def bits_of(spike):
    """Flatten a spike tensor to a python bit list in lane order."""
    return spike.reshape(-1).to(torch.int64).tolist()


def count_bits(count, width, entry):
    """Flatten a count tensor to the bit list of the packed count bus.

    Lane l occupies bits [l*width +: width] of o_out, so the lane counts are laid
    out low lane first, each one little-endian within its own field.
    """
    bits = []
    for value in count.reshape(-1).to(torch.int64).tolist():
        assert 0 <= value <= entry, f'count {value} outside [0, {entry}]'
        bits += [(value >> index) & 1 for index in range(width)]
    return bits


class capture:
    """Forward hook that keeps the most recent output of an encoder."""


    def __init__(self, module):
        self.spike = None
        module.register_forward_hook(self)


    def __call__(self, module, args, output):
        self.spike = output


class arm:
    """One polarity's input encoder plus its three conv_pc geometries."""


    def __init__(self, polarity):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 2}
        require_seeded_sys(codec, config)
        self.config = config
        self.weight = grid(OUT_CHANNELS * K, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity, 3)
        self.bias = grid(OUT_CHANNELS, (OUT_CHANNELS,), polarity, 1)
        self.values = [grid(BATCH * IN_CHANNELS * IN_H * IN_W, SHAPE, polarity, offset)
                       for offset in (0, 5)]
        self.enc = encode(codec)
        self.layers = [conv_pc(self.weight, self.bias if has_bias else None,
                               stride=stride, padding=padding, dilation=dilation, config=config)
                       for _, padding, stride, dilation, has_bias in GEOMETRY]
        self.weight_spike = [capture(layer.w_encoder) for layer in self.layers]
        self.bias_spike = [capture(layer.b_encoder) for layer in self.layers if layer.has_bias]
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Every elaborated geometry is translated: they differ in
        padding, stride, dilation, and bias, which change LANES and COUNT_W.
        """
        for (_, padding, stride, dilation, has_bias), layer in zip(GEOMETRY, self.layers):
            node = {"class": "conv_pc",
                    "config": {"weight": self.weight,
                               "bias": self.bias if has_bias else None,
                               "stride": stride, "padding": padding, "dilation": dilation,
                               "config": self.config,
                               "lanes": lanes_of(padding, stride, dilation)},
                    "inputs": {"input_spike": {"shape": SHAPE}}}
            expected = {"BATCH": BATCH, "IN_CHANNELS": IN_CHANNELS, "IN_H": IN_H, "IN_W": IN_W,
                        "OUT_CHANNELS": OUT_CHANNELS, "KERNEL_H": KERNEL[0], "KERNEL_W": KERNEL[1],
                        "STRIDE": stride, "PADDING": padding, "DILATION": dilation,
                        "HAS_BIAS": int(has_bias), "COUNT_W": count_width(has_bias),
                        "LANES": lanes_of(padding, stride, dilation)}
            binding = translate_node(node)
            assert binding.rtl_module == f"conv_pc_{self.polarity}", binding.rtl_module
            assert binding.parameters == expected, \
                f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"


    def reset(self):
        """Restart the input encoder and every layer."""
        self.enc.reset()
        for layer in self.layers:
            layer.reset()


    def step(self, index):
        """Advance one timestep over input tensor `index`; return the row columns."""
        spike = self.enc(self.values[index])
        outputs = [layer(spike) for layer in self.layers]
        weight_bits = bits_of(self.weight_spike[0].spike)
        for other in self.weight_spike[1:]:
            assert weight_bits == bits_of(other.spike), \
                'the layers encoded different weight spikes'
        bias_bits = bits_of(self.bias_spike[0].spike)
        for other in self.bias_spike[1:]:
            assert bias_bits == bits_of(other.spike), \
                'the layers encoded different bias spikes'
        pad_bits = [self.pad_bit(layer) for layer in self.layers if hasattr(layer, 'pad_bits')]
        assert len(set(pad_bits)) <= 1, 'the layers used different pad spikes'
        pad = pad_bits[0] if pad_bits else 0
        counts = [count_bits(out, count_width(has_bias), K + int(has_bias))
                  for out, (_, _, _, _, has_bias) in zip(outputs, GEOMETRY)]
        return bits_of(spike), weight_bits, bias_bits, [pad], counts


    @staticmethod
    def pad_bit(layer):
        """The pad spike the layer just consumed, from its own pad sequence."""
        return int(layer.pad_bits[(layer.timestep_cur - 1) % layer.pad_len])


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps over the other input tensor run before the recorded
    sequence and are followed by reset(), so the recorded block starts from
    encoders that had advanced. Its outputs must equal the same block recorded
    from fresh models. The RTL holds no state, so this checks the Python side
    the vectors come from rather than an RTL reset.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    for one in arms:
        one.check_mapping()
    if dirty:
        for _ in range(dirty):
            for one in arms:
                one.step(1 - index)
        for one in arms:
            one.reset()

    for _ in range(TIMESTEPS):
        inputs = []
        weights = []
        biases = []
        pads = []
        outputs = []
        for one in arms:
            in_bits, weight_bits, bias_bits, pad_bits, out_bits = one.step(index)
            inputs.append(as_binary(in_bits))
            weights.append(as_binary(weight_bits))
            biases.append(as_binary(bias_bits))
            if one.polarity == 'bipolar':
                pads.append(as_binary(pad_bits))
            outputs.append([as_binary(bits) for bits in out_bits])
        rows.append(" ".join(inputs + weights + biases + pads + outputs[0] + outputs[1]))
    return arms[0].layers[0].hw.pp_delay


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["in_u", "in_b", "w_u", "w_b", "bias_u", "bias_b", "pad_b"]
    header += [f"out_{polarity}_{label}" for polarity in "ub" for label, *_ in GEOMETRY]
    rows = [" ".join(header)]
    pp_delay = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=TIMESTEPS // 2)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:], \
        'reset() did not restart the encoders at their opening state'
    # Bipolar p = (x + 1) / 2 over the unipolar grid would reproduce the unipolar
    # columns exactly, leaving the bipolar arm no distinct stimulus.
    columns = [row.split() for row in rows[1:]]
    assert [row[0] for row in columns] != [row[1] for row in columns], \
        'bipolar input stimulus is identical to unipolar'
    assert [row[2] for row in columns] != [row[3] for row in columns], \
        'bipolar weight stimulus is identical to unipolar'

    defines = [("BATCH", BATCH), ("IN_CHANNELS", IN_CHANNELS), ("IN_H", IN_H), ("IN_W", IN_W),
               ("OUT_CHANNELS", OUT_CHANNELS), ("KERNEL_H", KERNEL[0]), ("KERNEL_W", KERNEL[1]),
               ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    for label, padding, stride, dilation, has_bias in GEOMETRY:
        suffix = label.upper()
        defines += [(f"PADDING_{suffix}", padding), (f"STRIDE_{suffix}", stride),
                    (f"DILATION_{suffix}", dilation), (f"HAS_BIAS_{suffix}", int(has_bias)),
                    (f"COUNT_W_{suffix}", count_width(has_bias)),
                    (f"LANES_{suffix}", lanes_of(padding, stride, dilation))]
    PARAMS.write_text("".join(f"`define GEN_{name} {value}\n" for name, value in defines))
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} ("
        + ", ".join(f"GEN_{name}={value}" for name, value in defines)
        + ")"
    )


if __name__ == "__main__":
    main()
