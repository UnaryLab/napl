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
tensor itself. A tap that falls outside the input reads the pad port on the bipolar
variant, whose bit the model draws from a separate rate-0.5 encoder on its own
sobol dimension (a deterministic toggle would correlate with the weight stream).
The unipolar variant has no pad port: its zero pad is a constant zero spike, so
only `pad_b` is a column.

The model re-encodes the weight, the bias, and the pad stream from externally
held tensors every timestep, so all three arrive on RTL ports rather than being
rebuilt in hardware. The weight and bias spikes are captured from the model's own
encoders with forward hooks, so the columns are the exact bits the model reduced;
the pad bit is read from the model's own precomputed pad sequence at the index
the model used this timestep (the pad encoder is not called in forward, so a hook
cannot see it).

Eight DUT configurations share each row, one column of expected output each: four
geometries per polarity, chosen so padding, bias, stride, and dilation each vary.

  a  -- padding 1, stride 1, dilation 1, with bias  (pad stream, entry = K + 1)
  b  -- padding 0, stride 1, dilation 1, no bias    (no padded taps, entry = K)
  c  -- padding 2, stride 2, dilation 2, with bias  (strided, dilated, padded)
  d  -- c's geometry at an explicit scale below its fan-in

a, b and c take the default scale, so they resolve the mapping's fan-in branch;
d takes an explicit one, which resolves the other branch and is also what lets an
accumulator move at all: with scale == entry the bipolar offset
(entry - scale) / 2 is 0 and a carry subtracts the whole entry, so the
accumulator stays inside [0, entry - 1]. Geometry d is what carries the positive
and the negative saturation blocks, which make WIDTH and the negative clamp
observable.

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded streams. Three sequences follow
each other: one over the first input tensor, one over the second, then the first
replayed after the second has dirtied the accumulators and reset() has cleared
them, which must reproduce the first sequence bit for bit.

Output: ../vec/conv.vec, one line per timestep:

    <rst> <in_u> <in_b> <w_u> <w_b> <bias_u> <bias_b> <pad_b>
    <out_u_a> <out_u_b> <out_u_c> <out_u_d> <out_b_a> <out_b_b> <out_b_c> <out_b_d>

`in_*` is BATCH*IN_CHANNELS*IN_H*IN_W binary digits, MSB first, so input element
(b, ic, ih, iw) occupies i_input_spike[((b*IN_CHANNELS + ic)*IN_H + ih)*IN_W + iw].
`w_*` is OUT_CHANNELS*K digits with out channel oc, tap t at i_weight[oc*K + t],
the tap order being (in_channel, kernel_row, kernel_col) row-major. `bias_*` is
OUT_CHANNELS digits, `pad_b` one digit, and each `out_*` column is that
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

# A divisor other than the fan-in, so geometry d resolves SCALE through the
# explicit-scale branch of the mapping expression and its accumulator charges
# under the saturation block.
SCALE_D = 9
# Negative-clamp block: an unpadded bipolar lane of geometry d falls by the
# offset (ENTRY - SCALE_D) / 2 = 5 per cycle with its popcount at zero, so
# 2**(WIDTH-1) / 5 = 410 cycles reach the clamp; the charge back up runs at
# ENTRY - offset = 14 per cycle and needs 2**(WIDTH-1) / 14 = 147 to break silence.
NEG_DRAIN = 460
NEG_CHARGE = 200

# label, padding, stride, dilation, has_bias, explicit scale (None = fan-in)
GEOMETRY = [
    ("a", 1, 1, 1, True, None),
    ("b", 0, 1, 1, False, None),
    ("c", 2, 2, 2, True, None),
    ("d", 2, 2, 2, True, SCALE_D),
]


def scale_of(has_bias, scale):
    """The output divisor one geometry elaborates."""
    return K + int(has_bias) if scale is None else scale


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


class capture:
    """Forward hook that keeps the most recent output of an encoder."""


    def __init__(self, module):
        self.spike = None
        module.register_forward_hook(self)


    def __call__(self, module, args, output):
        self.spike = output


class arm:
    """One polarity's input encoder plus its four conv geometries."""


    def __init__(self, polarity, rail=None):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'dim': 2, 'scale': None, 'width': ACC_WIDTH}
        require_seeded_sys(codec, config)
        self.config = config
        self.configs = [dict(config, scale=scale) for *_, scale in GEOMETRY]
        require_seeded_sys(*self.configs)
        if rail is not None:
            # rail='pos' rails the bias with the weight, so the low input block
            # still carries the bias addend; rail='neg' rails it low, which is
            # what takes an interior lane's popcount to zero and lets the bipolar
            # offset drive the accumulator onto its negative clamp.
            low, high = RANGE[polarity][0], 1.0
            self.weight = torch.full((OUT_CHANNELS, IN_CHANNELS) + KERNEL, high,
                                     dtype=global_config.ntype)
            self.bias = torch.full((OUT_CHANNELS,), high if rail == 'pos' else low,
                                   dtype=global_config.ntype)
            self.values = [torch.full(SHAPE, value, dtype=global_config.ntype)
                           for value in (low, high)]
        else:
            self.weight = grid(OUT_CHANNELS * K, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity, 3)
            self.bias = grid(OUT_CHANNELS, (OUT_CHANNELS,), polarity, 1)
            self.values = [grid(BATCH * IN_CHANNELS * IN_H * IN_W, SHAPE, polarity, offset)
                           for offset in (0, 5)]
        self.enc = encode(codec)
        self.layers = [conv(self.weight, self.bias if has_bias else None,
                            stride=stride, padding=padding, dilation=dilation,
                            config=layer_config)
                       for (_, padding, stride, dilation, has_bias, _), layer_config
                       in zip(GEOMETRY, self.configs)]
        self.weight_spike = [capture(layer.w_encoder) for layer in self.layers]
        self.bias_spike = [capture(layer.b_encoder) for layer in self.layers if layer.has_bias]
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Every elaborated geometry is translated: they differ in
        padding, stride, dilation, and bias, which change LANES, ENTRY, and so
        the default scale, and geometry d takes an explicit scale, which is the
        other branch of the SCALE expression.
        """
        for (_, padding, stride, dilation, has_bias, scale), layer_config in zip(
                GEOMETRY, self.configs):
            node = {"class": "conv",
                    "config": {"weight": self.weight,
                               "bias": self.bias if has_bias else None,
                               "stride": stride, "padding": padding, "dilation": dilation,
                               "config": layer_config,
                               "lanes": lanes_of(padding, stride, dilation)},
                    "inputs": {"input_spike": {"shape": SHAPE}}}
            expected = {"BATCH": BATCH, "IN_CHANNELS": IN_CHANNELS, "IN_H": IN_H, "IN_W": IN_W,
                        "OUT_CHANNELS": OUT_CHANNELS, "KERNEL_H": KERNEL[0], "KERNEL_W": KERNEL[1],
                        "STRIDE": stride, "PADDING": padding, "DILATION": dilation,
                        "WIDTH": ACC_WIDTH, "HAS_BIAS": int(has_bias),
                        "SCALE": scale_of(has_bias, scale),
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


def emit_row(rows, arms, index, first):
    """Step both arms over input tensor `index` and append their golden row."""
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
    columns = ([f"{1 if first else 0}"] + inputs + weights + biases + pads
               + outputs[0] + outputs[1])
    rows.append(" ".join(columns))


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
        emit_row(rows, arms, index, timestep == 0)
    return arms[0].layers[0].hw.pp_delay


def run_saturation(rows):
    """Append the positive-clamp block, which is what makes WIDTH observable.

    Weight and bias sit at the polarity's high rail, so an interior lane's
    popcount is ENTRY over the high input block and the bias alone over the low
    one. With SCALE_D below ENTRY the high block charges geometry d's
    accumulators onto their positive clamp at 2**(WIDTH-1) - 1 and the low block
    drains them: the stored charge is how many further cycles a lane keeps
    emitting, so a narrower WIDTH stops emitting sooner and its column differs.
    A geometry whose scale is its fan-in cannot charge at all, so d is the only
    column this block moves.
    """
    arms = [arm('unipolar', rail='pos'), arm('bipolar', rail='pos')]
    charge = [1] * (2 * TIMESTEPS) + [0] * (2 * TIMESTEPS)
    peak = [0, 0]
    for step, index in enumerate(charge):
        emit_row(rows, arms, index, step == 0)
        for slot, one in enumerate(arms):
            peak[slot] = max(peak[slot], int(one.layers[3].acc.accumulator.reshape(-1).max()))
    # One carry is subtracted after the clamp, so a clamped accumulator is
    # retained at acc_max - SCALE_D.
    for slot, one in enumerate(arms):
        acc_max = one.layers[3].acc.acc_max
        assert peak[slot] == acc_max - SCALE_D, \
            f'{one.polarity} geometry d peaked at {peak[slot]}, short of {acc_max}'


def run_negative(rows):
    """Append the negative-clamp block, which is what makes ACC_LO observable.

    The positive block cannot reach the negative clamp: its high rail bias keeps
    an addend on every cycle, and only a bipolar accumulator moves down at all,
    by the offset (ENTRY - SCALE) / 2 that a scale below the fan-in creates. So
    this block rails the bias low as well, which takes an unpadded lane's
    popcount to zero over the low input block, and holds it there long enough for
    geometry d's bipolar accumulator to fall onto -2**(WIDTH-1). The high input
    block then recharges it: how long a lane stays silent on the way back up is
    what the clamp sets, so a shallower ACC_LO starts emitting sooner and its
    column differs. A padded lane keeps the rate-0.5 pad addends and so drains
    more slowly, which is why the clamp is asserted on the deepest lane.

    The unipolar accumulator has no offset and its carry only ever subtracts down
    to zero, so its negative clamp is unreachable by construction, not for want
    of stimulus.
    """
    arms = [arm('unipolar', rail='neg'), arm('bipolar', rail='neg')]
    drain = [0] * NEG_DRAIN + [1] * NEG_CHARGE
    bottom = [0, 0]
    for step, index in enumerate(drain):
        emit_row(rows, arms, index, step == 0)
        for slot, one in enumerate(arms):
            bottom[slot] = min(bottom[slot], int(one.layers[3].acc.accumulator.reshape(-1).min()))
    acc_min = arms[1].layers[3].acc.acc_min
    assert bottom[1] == acc_min, \
        f'bipolar geometry d bottomed at {bottom[1]}, short of {acc_min}'
    assert bottom[0] == 0, \
        f'unipolar geometry d went negative, to {bottom[0]}'


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["rst", "in_u", "in_b", "w_u", "w_b", "bias_u", "bias_b", "pad_b"]
    header += [f"out_{polarity}_{label}" for polarity in "ub" for label, *_ in GEOMETRY]
    rows = [" ".join(header)]
    pp_delay = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=TIMESTEPS // 2)
    run_saturation(rows)
    run_negative(rows)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:1 + 3 * TIMESTEPS], \
        'reset() did not restore the opening accumulator state'
    # Bipolar p = (x + 1) / 2 over the unipolar grid would reproduce the unipolar
    # columns exactly, leaving the bipolar arm no distinct stimulus.
    columns = [row.split() for row in rows[1:1 + 3 * TIMESTEPS]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        'bipolar input stimulus is identical to unipolar'
    assert [row[3] for row in columns] != [row[4] for row in columns], \
        'bipolar weight stimulus is identical to unipolar'

    defines = [("BATCH", BATCH), ("IN_CHANNELS", IN_CHANNELS), ("IN_H", IN_H), ("IN_W", IN_W),
               ("OUT_CHANNELS", OUT_CHANNELS), ("KERNEL_H", KERNEL[0]), ("KERNEL_W", KERNEL[1]),
               ("WIDTH", ACC_WIDTH), ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    for label, padding, stride, dilation, has_bias, scale in GEOMETRY:
        suffix = label.upper()
        defines += [(f"PADDING_{suffix}", padding), (f"STRIDE_{suffix}", stride),
                    (f"DILATION_{suffix}", dilation), (f"HAS_BIAS_{suffix}", int(has_bias)),
                    (f"SCALE_{suffix}", scale_of(has_bias, scale)),
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
