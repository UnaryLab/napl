"""
Generate golden test vectors for the conv_ugemm module RTL straight from napl's
functional Python model (napl.sim.module.conv_ugemm) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

conv_ugemm reduces one lane per output pixel-channel (batch, out_channel,
out_row, out_col). Its per-timestep partial sum is the plain count of the
conditionally generated product spikes over the im2col patch plus the optional
bias spike -- no Gaines rewrite is involved, because the model already sums 0/1
spikes -- and add_any popcounts its ENTRY-bit input. So the RTL lane is exactly

    mul_ugemm_<polarity>  -- one per (output position, patch tap) weight bit
    encode                -- one per output position, the bias bit
    add_any_<polarity>    -- one per output position, the scaled accumulator

all of them operation-layer circuits, instantiated rather than rebuilt.

Shared sequence indices: the model's patch spike broadcasts over the output
channels, so one sequence-index pair per (spatial position, tap) serves every
output channel. The RTL gives each lane its own pair, which is OUT_CHANNELS
copies of the model's one pair; mul_ugemm advances an index from i_input_0 alone,
and every copy sees the same patch spike, so the copies stay equal and the
outputs are bit-exact with the shared-index model.

Weight and bias are held fixed-point codes rather than spike streams (the model
sets internal_encode = True: it generates both streams itself), so they go to
../vec/conv_ugemm_operand_u.hex and _b.hex instead of vector columns. The pad bit
is likewise internal: bipolar zero padding is the alternating 0/1 stream the
model derives from its own timestep counter, which the RTL builds from a jkff
held at J = K = 1, and unipolar padding is a zero spike. A padded tap is not a
don't-care -- it advances that tap's sequence index -- so the padded geometries
below are what verify it.

Both `mul_ugemm_*` and `encode` $readmemb their number-sequence ROM from a path
relative to the simulation cwd (module/conv_ugemm/), so this file writes
vec/mul_ugemm_rom.hex and vec/encode_rom.hex with the sequence of the built
model. Both tables are the same sequence, which is why the bias comparator can be
the encode circuit: conv_ugemm indexes self.mul.num_seq with the free-running
timestep for the bias bit.

Sequence-index bound: mul_ugemm advances its indices without a modulo, so a run
stays valid for at most LEN = 2**ceil(log2(timestep)) enabling timesteps, and in
bipolar for at most LEN non-enabling ones as well. TIMESTEPS is pinned to LEN,
which satisfies both bounds however the input stream splits.

Six DUT configurations share each row, one column of expected output each: three
geometries per polarity, chosen so padding, bias, stride, and dilation each vary.

  a  -- padding 1, stride 1, dilation 1, with bias  (pad stream, entry = K + 1)
  b  -- padding 0, stride 1, dilation 1, no bias    (no padded taps, entry = K)
  c  -- padding 1, stride 2, dilation 2, with bias  (strided, dilated, padded)

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded input stream. Three sequences
follow each other: one over the first input tensor, one over the second, then the
first replayed after the second has dirtied the state and reset() has cleared it,
which must reproduce the first sequence bit for bit. The dirty block is an odd
number of timesteps, so the bipolar pad toggle and the bias comparator index are
both left at a value a missing reset would carry into the replay.

Output: ../vec/conv_ugemm.vec, one line per timestep:

    <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c> <out_b_a> <out_b_b> <out_b_c>

`in_*` is BATCH*IN_CHANNELS*IN_H*IN_W binary digits, MSB first, so input element
(b, ic, ih, iw) occupies i_input_spike[((b*IN_CHANNELS + ic)*IN_H + ih)*IN_W + iw].
Each `out_*` column is that configuration's LANES digits with lane (b, oc, oh, ow)
row-major.

Sizing values come from this file only and are emitted into
../vec/conv_ugemm_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay and the
row count, which the testbench compares against the rows it consumed.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_conv_ugemm.py
"""
import math
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import conv_ugemm
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "conv_ugemm.vec"
PARAMS = VEC_DIR / "conv_ugemm_params.vh"

# Shapes and codec mirror tests/module/test_conv_ugemm.py's fidelity-scale input,
# with the channel and kernel counts kept small: every lane elaborates K
# mul_ugemm cells, and each of those holds its own copy of the sequence ROM.
SHAPE = (1, 1, 4, 4)          # batch, in_channels, height, width
OUT_CHANNELS = 2
KERNEL = (2, 2)
TIMESTEP = 256
ACC_WIDTH = 12

BATCH, IN_CHANNELS, IN_H, IN_W = SHAPE
K = IN_CHANNELS * KERNEL[0] * KERNEL[1]

SEQ_WIDTH = math.ceil(math.log2(TIMESTEP))
LEN = 2 ** SEQ_WIDTH
# mul_ugemm's sequence indices have no modulo, so a run may not exceed LEN steps.
TIMESTEPS = min(TIMESTEP, LEN)
# An odd dirty block leaves the pad toggle and the bias index off their reset
# values, so a reset that misses either shows up in the replayed block.
DIRTY = TIMESTEPS // 2 + 1

# label, padding, stride, dilation, has_bias
GEOMETRY = [
    ("a", 1, 1, 1, True),
    ("b", 0, 1, 1, False),
    ("c", 1, 2, 2, True),
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


def operand_codes(count, stride, offset):
    """Return `count` fixed-point codes in [0, LEN] on the 1/LEN operand grid."""
    return [(i * stride + offset) % (LEN + 1) for i in range(count)]


WEIGHT_CODES = operand_codes(OUT_CHANNELS * K, 29, 5)
BIAS_CODES = operand_codes(OUT_CHANNELS, 53, 7)
INPUT_CODES = [operand_codes(BATCH * IN_CHANNELS * IN_H * IN_W, 71, 13),
               operand_codes(BATCH * IN_CHANNELS * IN_H * IN_W, 37, 101)]


def to_value(code, polarity):
    """Map a fixed-point probability code to the polarity's real value."""
    prob = code / LEN
    return prob if polarity == 'unipolar' else 2.0 * prob - 1.0


def tensor(codes, shape, polarity):
    """Build a model tensor from operand codes, in the polarity's value domain."""
    values = [to_value(code, polarity) for code in codes]
    return torch.tensor(values, dtype=global_config.ntype).reshape(shape)


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
    """One polarity's input encoder plus its three conv_ugemm geometries."""


    def __init__(self, polarity):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 2}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'scale': None, 'width': ACC_WIDTH}
        require_seeded_sys(codec, config)
        self.config = config
        self.weight = tensor(WEIGHT_CODES, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity)
        self.bias = tensor(BIAS_CODES, (OUT_CHANNELS,), polarity)
        self.values = [tensor(codes, SHAPE, polarity) for codes in INPUT_CODES]
        self.enc = encode(codec)
        # The input spikes the layers reduce are captured from the encoder's own
        # forward, so the golden column is provably the bits the model saw.
        self.input_spike = capture(self.enc)
        self.layers = [conv_ugemm(self.weight, self.bias if has_bias else None,
                                  stride=stride, padding=padding, dilation=dilation,
                                  config=config)
                       for _, padding, stride, dilation, has_bias in GEOMETRY]
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
            node = {"class": "conv_ugemm",
                    "config": {"weight": self.weight,
                               "bias": self.bias if has_bias else None,
                               "stride": stride, "padding": padding, "dilation": dilation,
                               "config": self.config,
                               "lanes": lanes_of(padding, stride, dilation)},
                    "inputs": {"input_spike": {"shape": SHAPE}}}
            expected = {"BATCH": BATCH, "IN_CHANNELS": IN_CHANNELS, "IN_H": IN_H, "IN_W": IN_W,
                        "OUT_CHANNELS": OUT_CHANNELS, "KERNEL_H": KERNEL[0], "KERNEL_W": KERNEL[1],
                        "STRIDE": stride, "PADDING": padding, "DILATION": dilation,
                        "SEQ_WIDTH": SEQ_WIDTH, "WIDTH": ACC_WIDTH,
                        "HAS_BIAS": int(has_bias), "SCALE": K + int(has_bias),
                        "LANES": lanes_of(padding, stride, dilation)}
            binding = translate_node(node)
            assert binding.rtl_module == f"conv_ugemm_{self.polarity}", binding.rtl_module
            assert binding.parameters == expected, \
                f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"
            assert layer.entry == K + int(has_bias), layer.entry


    def reset(self):
        """Restart the input encoder and every layer."""
        self.enc.reset()
        for layer in self.layers:
            layer.reset()


    def step(self, index):
        """Advance one timestep over input tensor `index`; return the row columns."""
        spike = self.enc(self.values[index])
        outputs = [layer(spike) for layer in self.layers]
        return bits_of(self.input_spike.spike), [bits_of(out) for out in outputs]


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps over the other input tensor run before the recorded
    sequence and are followed by reset(), so the recorded block starts from
    sequence indices, bias and pad counters, and accumulators that held state.
    Its outputs must equal the same block recorded from fresh models, which the
    testbench checks by pulsing i_rst_n on the opening rst=1 row.
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
        outputs = []
        for one in arms:
            in_bits, out_bits = one.step(index)
            inputs.append(as_binary(in_bits))
            outputs.append([as_binary(bits) for bits in out_bits])
        columns = [f"{1 if timestep == 0 else 0}"] + inputs + outputs[0] + outputs[1]
        rows.append(" ".join(columns))
    return arms[0].layers[0]


def write_rom(layer):
    """Emit the model's number-sequence ROM under both names the RTL reads.

    mul_ugemm_* reads vec/mul_ugemm_rom.hex and encode reads vec/encode_rom.hex,
    both relative to the simulation cwd. The layer drives the bias comparator from
    the same self.mul.num_seq table, so the two files hold identical lines.
    """
    num_seq = layer.mul.num_seq.detach().float().reshape(-1)
    assert num_seq.numel() == LEN, f'num_seq length {num_seq.numel()} != LEN {LEN}'
    lines = []
    for index in range(LEN):
        scaled = num_seq[index].item() * LEN
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f'num_seq[{index}] off the 1/{LEN} grid'
        lines.append(f"{code:0{SEQ_WIDTH}b}")
    text = "\n".join(lines) + "\n"
    (VEC_DIR / "mul_ugemm_rom.hex").write_text(text)
    (VEC_DIR / "encode_rom.hex").write_text(text)


def write_operands():
    """Emit the held operand codes: OUT_CHANNELS*K weights, then OUT_CHANNELS biases.

    An operand code is the probability code both polarities compare against, so
    one table serves the unipolar and the bipolar DUTs.
    """
    codes = list(WEIGHT_CODES) + list(BIAS_CODES)
    (VEC_DIR / "conv_ugemm_operand.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in codes) + "\n"
    )


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["rst", "in_u", "in_b"]
    header += [f"out_{polarity}_{label}" for polarity in "ub" for label, *_ in GEOMETRY]
    rows = [" ".join(header)]
    layer = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=DIRTY)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:], \
        'reset() did not restore the opening sequence-index, counter and accumulator state'

    write_rom(layer)
    write_operands()
    pp_delay = layer.hw.pp_delay
    defines = [("BATCH", BATCH), ("IN_CHANNELS", IN_CHANNELS), ("IN_H", IN_H), ("IN_W", IN_W),
               ("OUT_CHANNELS", OUT_CHANNELS), ("KERNEL_H", KERNEL[0]), ("KERNEL_W", KERNEL[1]),
               ("SEQ_WIDTH", SEQ_WIDTH), ("WIDTH", ACC_WIDTH),
               ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    for label, padding, stride, dilation, has_bias in GEOMETRY:
        suffix = label.upper()
        defines += [(f"PADDING_{suffix}", padding), (f"STRIDE_{suffix}", stride),
                    (f"DILATION_{suffix}", dilation), (f"HAS_BIAS_{suffix}", int(has_bias)),
                    (f"SCALE_{suffix}", K + int(has_bias)),
                    (f"LANES_{suffix}", lanes_of(padding, stride, dilation))]
    # The testbench sizes its scan tokens from the first geometry's lane count,
    # which only rejects an over-long column while it is the widest column here.
    widest = max([BATCH * IN_CHANNELS * IN_H * IN_W]
                 + [lanes_of(padding, stride, dilation)
                    for _, padding, stride, dilation, _ in GEOMETRY])
    assert widest == lanes_of(*GEOMETRY[0][1:4]), \
        f'geometry a is not the widest column ({widest}); the testbench MAX_CHARS assumes it is'
    PARAMS.write_text("".join(f"`define GEN_{name} {value}\n" for name, value in defines))
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} ("
        + ", ".join(f"GEN_{name}={value}" for name, value in defines)
        + ")"
    )


if __name__ == "__main__":
    main()
