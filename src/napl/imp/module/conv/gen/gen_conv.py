"""
Generate golden test vectors for the conv module RTL straight from napl's
functional Python model (napl.sim.module.conv) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

conv reduces one lane per output pixel-channel (batch, out_channel, out_row,
out_col). Its per-timestep partial sum over the im2col patch is a popcount of
Gaines products, exactly as in the linear module with the patch in place of the
input features:

    unipolar   psum = sum_k x_k * w_k                      = popcount(AND(x, w))
    bipolar    psum = 2*sum_k x_k*w_k - sum_k x_k
                      - sum_k w_k + K
                    = sum_k (1 - x_k - w_k + 2*x_k*w_k)    = popcount(XNOR(x, w))

with the bias spike added as one more addend. add_any popcounts its ENTRY-bit
input and takes the same ENTRY for its bipolar offset, so the RTL lane is
K = in_channels*kernel_h*kernel_w mul_gaines_<polarity> cells plus the optional
bias bit feeding one add_any_<polarity>:

    mul_gaines_<polarity>  -- one per (output position, patch tap)
    add_any_<polarity>     -- one per output position, the scaled accumulator

The im2col patch is wiring in the RTL, so the input port carries the NCHW spike
tensor itself. A tap that falls outside the input reads the pad port. For a
bipolar stream with padding the model draws that bit from a separate rate-0.5
encoder on its own sobol dimension (a deterministic toggle would correlate with
the weight stream); for a unipolar stream the pad is a zero spike.

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
replayed after the second has dirtied the accumulators and reset() has cleared
them, which must reproduce the first sequence bit for bit.

Output: ../vec/conv.vec, one line per timestep:

    <rst> <in_u> <in_b> <w_u> <w_b> <bias_u> <bias_b> <pad_u> <pad_b>
    <out_u_a> <out_u_b> <out_u_c> <out_b_a> <out_b_b> <out_b_c>

`in_*` is BATCH*IN_CHANNELS*IN_H*IN_W binary digits, MSB first, so input element
(b, ic, ih, iw) occupies i_input_spike[((b*IN_CHANNELS + ic)*IN_H + ih)*IN_W + iw].
`w_*` is OUT_CHANNELS*K digits with out channel oc, tap t at i_weight[oc*K + t],
the tap order being (in_channel, kernel_row, kernel_col) row-major. `bias_*` is
OUT_CHANNELS digits, `pad_*` one digit, and each `out_*` column is that
configuration's LANES digits with lane (b, oc, oh, ow) row-major.

Sizing values come from this file only and are emitted into
../vec/conv_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay and the
row count, which the testbench compares against the rows it consumed.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_conv.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import conv
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "conv.vec"
PARAMS = VEC_DIR / "conv_params.vh"

# Shapes and codec mirror tests/module/test_conv.py's fidelity-scale checks,
# scaled down to keep the elaborated lane count small.
SHAPE = (1, 2, 6, 6)          # batch, in_channels, height, width
OUT_CHANNELS = 3
KERNEL = (3, 3)
TIMESTEP = 256
ACC_WIDTH = 12
TIMESTEPS = TIMESTEP

BATCH, IN_CHANNELS, IN_H, IN_W = SHAPE
K = IN_CHANNELS * KERNEL[0] * KERNEL[1]

# label, padding, stride, dilation, has_bias
GEOMETRY = [
    ("a", 1, 1, 1, True),
    ("b", 0, 1, 1, False),
    ("c", 2, 2, 2, True),
]


def out_hw(padding, stride, dilation):
    """Output height and width of one geometry, as the RTL derives them."""
    height = (IN_H + 2 * padding - dilation * (KERNEL[0] - 1) - 1) // stride + 1
    width = (IN_W + 2 * padding - dilation * (KERNEL[1] - 1) - 1) // stride + 1
    return height, width


def lanes_of(padding, stride, dilation):
    """Output positions one geometry elaborates."""
    height, width = out_hw(padding, stride, dilation)
    return BATCH * OUT_CHANNELS * height * width


def grid(count, shape, polarity, offset):
    """A deterministic operand tensor spanning the polarity's value range."""
    low = 0.0 if polarity == 'unipolar' else -1.0
    steps = torch.linspace(low, 1.0, count, dtype=global_config.ntype)
    return steps.roll(offset).reshape(shape)


def as_binary(bits):
    """Render a bit list as a Verilog %b string, highest bit index first."""
    return "".join(str(int(bit)) for bit in reversed(bits))


def bits_of(spike):
    """Flatten a spike tensor to a python bit list in lane order."""
    return spike.reshape(-1).to(torch.int64).tolist()


class capture:
    """Forward hook that keeps the most recent output of an encoder."""


    def __init__(self, module):
        self.spike = None
        module.register_forward_hook(self)


    def __call__(self, module, args, output):
        self.spike = output


class arm:
    """One polarity's input encoder plus its three conv geometries."""


    def __init__(self, polarity):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'dim': 2, 'scale': None, 'width': ACC_WIDTH}
        require_seeded_sys(codec, config)
        self.config = config
        self.weight = grid(OUT_CHANNELS * K, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity, 3)
        self.bias = grid(OUT_CHANNELS, (OUT_CHANNELS,), polarity, 1)
        self.values = [grid(BATCH * IN_CHANNELS * IN_H * IN_W, SHAPE, polarity, offset)
                       for offset in (0, 5)]
        self.enc = encode(codec)
        self.layers = [conv(self.weight, self.bias if has_bias else None,
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
        padding, stride, dilation, and bias, which change LANES, ENTRY, and so
        the default scale.
        """
        for (_, padding, stride, dilation, has_bias), layer in zip(GEOMETRY, self.layers):
            node = {"class": "conv",
                    "config": {"weight": self.weight,
                               "bias": self.bias if has_bias else None,
                               "stride": stride, "padding": padding, "dilation": dilation,
                               "config": self.config,
                               "lanes": lanes_of(padding, stride, dilation)},
                    "inputs": {"input_spike": {"shape": SHAPE}}}
            expected = {"BATCH": BATCH, "IN_CHANNELS": IN_CHANNELS, "IN_H": IN_H, "IN_W": IN_W,
                        "OUT_CHANNELS": OUT_CHANNELS, "KERNEL_H": KERNEL[0], "KERNEL_W": KERNEL[1],
                        "STRIDE": stride, "PADDING": padding, "DILATION": dilation,
                        "WIDTH": ACC_WIDTH, "HAS_BIAS": int(has_bias),
                        "SCALE": K + int(has_bias),
                        "LANES": lanes_of(padding, stride, dilation)}
            binding = translate_node(node)
            assert binding.rtl_module == f"conv_{self.polarity}", binding.rtl_module
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
        return (bits_of(spike), weight_bits, bias_bits, [pad],
                [bits_of(out) for out in outputs])


    @staticmethod
    def pad_bit(layer):
        """The pad spike the layer just consumed, from its own pad sequence."""
        return int(layer.pad_bits[(layer.timestep_cur - 1) % layer.pad_len])


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps over the other input tensor run before the recorded
    sequence and are followed by reset(), so the recorded block starts from
    accumulators that held state. Its outputs must equal the same block recorded
    from fresh models, which the testbench checks by pulsing i_rst_n on the
    opening rst=1 row.
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

    for timestep in range(TIMESTEPS):
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
            pads.append(as_binary(pad_bits))
            outputs.append([as_binary(bits) for bits in out_bits])
        columns = ([f"{1 if timestep == 0 else 0}"] + inputs + weights + biases + pads
                   + outputs[0] + outputs[1])
        rows.append(" ".join(columns))
    return arms[0].layers[0].hw.pp_delay


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["rst", "in_u", "in_b", "w_u", "w_b", "bias_u", "bias_b", "pad_u", "pad_b"]
    header += [f"out_{polarity}_{label}" for polarity in "ub" for label, *_ in GEOMETRY]
    rows = [" ".join(header)]
    pp_delay = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=TIMESTEPS // 2)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:], \
        'reset() did not restore the opening accumulator state'

    defines = [("BATCH", BATCH), ("IN_CHANNELS", IN_CHANNELS), ("IN_H", IN_H), ("IN_W", IN_W),
               ("OUT_CHANNELS", OUT_CHANNELS), ("KERNEL_H", KERNEL[0]), ("KERNEL_W", KERNEL[1]),
               ("WIDTH", ACC_WIDTH), ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    for label, padding, stride, dilation, has_bias in GEOMETRY:
        suffix = label.upper()
        defines += [(f"PADDING_{suffix}", padding), (f"STRIDE_{suffix}", stride),
                    (f"DILATION_{suffix}", dilation), (f"HAS_BIAS_{suffix}", int(has_bias)),
                    (f"SCALE_{suffix}", K + int(has_bias)),
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
