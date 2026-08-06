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
../vec/conv_ugemm_operand.hex instead of vector columns. One file serves both
polarity DUTs: an operand code is the probability code either polarity compares
against. Geometry d reads ../vec/conv_ugemm_operand_d.hex, whose weights sit at
the top code and whose bias sits at 0, so a railed input gives its lanes a
partial sum of K and a silent one gives 0. The pad bit
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

Sequence-index bound: mul_ugemm advances its indices without a modulo and reads
the sequence on every timestep, so a run stays valid for at most
LEN = 2**ceil(log2(timestep)) enabling timesteps, and in bipolar for at most LEN
non-enabling ones as well. A run that uses all LEN enabling timesteps has no
cycle left, so a block that charges an accumulator and then drains it holds at
most LEN - 1 of each. TIMESTEPS is pinned to LEN, which satisfies both bounds
however the input stream splits.

Eight DUT configurations share each row, one column of expected output each: four
geometries per polarity, chosen so padding, bias, stride, and dilation each vary.

  a  -- padding 1, stride 1, dilation 1, with bias  (pad stream, entry = K + 1)
  b  -- padding 0, stride 1, dilation 1, no bias    (no padded taps, entry = K)
  c  -- padding 1, stride 2, dilation 2, with bias  (strided, dilated, padded)
  d  -- b's geometry with a bias, at an explicit scale below its fan-in

a, b and c take the default scale, so they resolve the mapping's fan-in branch;
d takes an explicit one, which resolves the other branch and is also what lets an
accumulator move at all: with scale == entry the bipolar offset
(entry - scale) / 2 is 0 and a carry subtracts the whole entry, so the stored
value stays inside [0, entry - 1]. Geometry d holds railed operands and carries
the saturation block, which is what makes WIDTH observable.

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded input stream. Three sequences
follow each other: one over the first input tensor, one over the second, then the
first replayed after the second has dirtied the state and reset() has cleared it,
which must reproduce the first sequence bit for bit. The dirty block is an odd
number of timesteps, so the bipolar pad toggle and the bias comparator index are
both left at a value a missing reset would carry into the replay.

Output: ../vec/conv_ugemm.vec, one line per timestep:

    <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c> <out_u_d>
    <out_b_a> <out_b_b> <out_b_c> <out_b_d>

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
SHAPE = (1, 3, 4, 4)          # batch, in_channels, height, width
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

# A divisor below the fan-in for geometry d, so it resolves SCALE through the
# explicit-scale branch of the mapping expression and its accumulator charges
# under the saturation block. With the bias code at 0 a railed lane's partial sum
# is K, so the unipolar accumulator climbs at K - SCALE_D = 6 per cycle: the
# pre-carry sum reaches 6 * SAT_CHARGE + SCALE_D = 1086, past the 1023 clamp of
# an 11-bit accumulator and short of the 2047 of the elaborated 12. The drain
# then falls at SCALE_D per cycle, so the clamped arm runs out of charge 10
# cycles before the elaborated one and their columns part. Both blocks stay
# inside the LEN - 1 spiking and LEN - 1 silent budget the sequence indices set.
SCALE_D = 6
SAT_CHARGE = 180
SAT_DRAIN = 190
# Stored excursions geometry d's accumulators reach under that block, as measured
# from the model. RULE_IMP.md and reports/napl-gen-rtl-report.md quote them, and
# run_saturation asserts each exactly, so a quoted figure that drifts from the
# model fails the generator instead of standing uncontradicted.
SAT_PEAK_UNI = 1080.0
SAT_PEAK_BI = 450.0
SAT_TROUGH_BI = -497.0
# Rails as model values: probability 1 and 0 in the polarity's own domain.
RAIL_VALUE = {'unipolar': (1.0, 0.0), 'bipolar': (1.0, -1.0)}

# label, padding, stride, dilation, has_bias, explicit scale (None = fan-in)
GEOMETRY = [
    ("a", 1, 1, 1, True, None),
    ("b", 0, 1, 1, False, None),
    ("c", 1, 2, 2, True, None),
    ("d", 0, 1, 1, True, SCALE_D),
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


def operand_codes(count, stride, offset):
    """Return `count` fixed-point codes in [0, LEN] on the 1/LEN operand grid."""
    return [(i * stride + offset) % (LEN + 1) for i in range(count)]


WEIGHT_CODES = operand_codes(OUT_CHANNELS * K, 29, 5)
BIAS_CODES = operand_codes(OUT_CHANNELS, 53, 7)
# Geometry d's railed operands: every weight at the top code, so a unipolar
# product spike is the input spike itself and a railed lane counts K, and the
# bias at 0, so it never spikes and a silent lane counts 0.
WEIGHT_CODES_D = [LEN] * (OUT_CHANNELS * K)
BIAS_CODES_D = [0] * OUT_CHANNELS
# One code pair per polarity: a shared pair would put the same probability grid
# through both encoders, so the bipolar input column would copy the unipolar one.
# The weight and bias codes stay shared, since one operand file feeds both DUTs.
INPUT_CODES = {
    'unipolar': [operand_codes(BATCH * IN_CHANNELS * IN_H * IN_W, 71, 13),
                 operand_codes(BATCH * IN_CHANNELS * IN_H * IN_W, 37, 101)],
    'bipolar': [operand_codes(BATCH * IN_CHANNELS * IN_H * IN_W, 43, 97),
                operand_codes(BATCH * IN_CHANNELS * IN_H * IN_W, 59, 151)],
}


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
    """One polarity's input encoder plus its four conv_ugemm geometries."""


    def __init__(self, polarity):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 2}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'scale': None, 'width': ACC_WIDTH}
        require_seeded_sys(codec, config)
        self.config = config
        self.configs = [dict(config, scale=scale) for *_, scale in GEOMETRY]
        require_seeded_sys(*self.configs)
        self.weight = tensor(WEIGHT_CODES, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity)
        self.bias = tensor(BIAS_CODES, (OUT_CHANNELS,), polarity)
        self.weight_d = tensor(WEIGHT_CODES_D, (OUT_CHANNELS, IN_CHANNELS) + KERNEL, polarity)
        self.bias_d = tensor(BIAS_CODES_D, (OUT_CHANNELS,), polarity)
        self.values = [tensor(codes, SHAPE, polarity) for codes in INPUT_CODES[polarity]]
        self.enc = encode(codec)
        # The input spikes the layers reduce are captured from the encoder's own
        # forward, so the golden column is provably the bits the model saw.
        self.input_spike = capture(self.enc)
        self.layers = [
            conv_ugemm(self.weight_of(scale), self.bias_of(scale) if has_bias else None,
                       stride=stride, padding=padding, dilation=dilation, config=layer_config)
            for (_, padding, stride, dilation, has_bias, scale), layer_config
            in zip(GEOMETRY, self.configs)]
        # A shadow of geometry d one bit narrower. It is never emitted: the
        # saturation block requires its output to part from geometry d's, which
        # is what makes the elaborated WIDTH observable in a vector row.
        _, padding, stride, dilation, has_bias, _ = GEOMETRY[3]
        self.layer_narrow = conv_ugemm(self.weight_d, self.bias_d,
                                       stride=stride, padding=padding, dilation=dilation,
                                       config=dict(self.configs[3], width=ACC_WIDTH - 1))
        self.narrow_bits = None
        self.reset()


    def weight_of(self, scale):
        """The weight codes a geometry holds: railed for the scaled arm."""
        return self.weight if scale is None else self.weight_d


    def bias_of(self, scale):
        """The bias codes a geometry holds: railed to 0 for the scaled arm."""
        return self.bias if scale is None else self.bias_d


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Every elaborated geometry is translated: they differ in
        padding, stride, dilation, and bias, which change LANES, ENTRY, and so
        the default scale, and geometry d takes an explicit scale, which is the
        other branch of the SCALE expression.
        """
        for (_, padding, stride, dilation, has_bias, scale), layer_config, layer in zip(
                GEOMETRY, self.configs, self.layers):
            node = {"class": "conv_ugemm",
                    "config": {"weight": self.weight_of(scale),
                               "bias": self.bias_of(scale) if has_bias else None,
                               "stride": stride, "padding": padding, "dilation": dilation,
                               "config": layer_config,
                               "lanes": lanes_of(padding, stride, dilation)},
                    "inputs": {"input_spike": {"shape": SHAPE}}}
            expected = {"BATCH": BATCH, "IN_CHANNELS": IN_CHANNELS, "IN_H": IN_H, "IN_W": IN_W,
                        "OUT_CHANNELS": OUT_CHANNELS, "KERNEL_H": KERNEL[0], "KERNEL_W": KERNEL[1],
                        "STRIDE": stride, "PADDING": padding, "DILATION": dilation,
                        "SEQ_WIDTH": SEQ_WIDTH, "WIDTH": ACC_WIDTH,
                        "HAS_BIAS": int(has_bias), "SCALE": scale_of(has_bias, scale),
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
        self.layer_narrow.reset()


    def step(self, index, rail=None):
        """Advance one timestep; return the row columns.

        `rail` drives every input element at the polarity's high or low rail
        instead of input tensor `index`, which is what charges and drains the
        scaled accumulator.
        """
        value = self.values[index]
        if rail is not None:
            level = RAIL_VALUE[self.polarity][0 if rail else 1]
            value = torch.full(SHAPE, level, dtype=global_config.ntype)
        spike = self.enc(value)
        outputs = [layer(spike) for layer in self.layers]
        self.narrow_bits = bits_of(self.layer_narrow(spike))
        return bits_of(self.input_spike.spike), [bits_of(out) for out in outputs]


def emit_row(rows, arms, index, first, rail=None):
    """Step both arms over input tensor `index` (or a rail) and append their row.

    Returns whether either arm's geometry d parted from its one-bit-narrower
    shadow on this cycle, which is the row a corrupted WIDTH would fail on.
    """
    inputs = []
    outputs = []
    parted = False
    for one in arms:
        in_bits, out_bits = one.step(index, rail)
        inputs.append(as_binary(in_bits))
        outputs.append([as_binary(bits) for bits in out_bits])
        parted = parted or out_bits[3] != one.narrow_bits
    columns = [f"{1 if first else 0}"] + inputs + outputs[0] + outputs[1]
    rows.append(" ".join(columns))
    return parted


def run_saturation(rows):
    """Append the saturation block, which is what makes WIDTH observable.

    Geometry d holds its weights at the top code and its bias at 0, so the
    railed input block gives every one of its lanes a partial sum of K and the
    silent block gives 0. With SCALE_D below ENTRY the railed block charges the
    unipolar accumulator past the clamp of the next narrower width and the silent
    block drains it: the stored charge is how many further cycles a lane keeps
    emitting, so a narrower WIDTH stops emitting sooner and its column differs.

    The bipolar accumulator moves at the offset (ENTRY - SCALE_D) / 2 = 3.5
    rather than at K - SCALE_D, which the sequence-index budget stops well inside
    both clamps, so its column is the same at either width. The assertions below
    record that reach rather than assume it.

    A unipolar silent cycle advances no sequence index, but every cycle reads the
    sequence, so the charge is capped at LEN - 1 spiking cycles; the bipolar arm
    advances its inverse index on each silent cycle, which caps the drain at
    LEN - 1 as well.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    rails = [True] * SAT_CHARGE + [False] * SAT_DRAIN
    peak = [0, 0]
    bottom = [0, 0]
    parted = False
    for step, rail in enumerate(rails):
        parted = emit_row(rows, arms, 0, step == 0, rail) or parted
        for slot, one in enumerate(arms):
            stored = one.layers[3].acc.accumulator.reshape(-1)
            peak[slot] = max(peak[slot], float(stored.max()))
            bottom[slot] = min(bottom[slot], float(stored.min()))
    assert parted, 'the saturation block never parted from a one-bit-narrower accumulator'
    acc_max = arms[0].layers[3].acc.acc_max
    acc_min = arms[0].layers[3].acc.acc_min
    narrower_max = 2 ** (ACC_WIDTH - 2) - 1
    narrower_min = -2 ** (ACC_WIDTH - 2)
    # The unipolar arm charges past the next narrower clamp, which is the
    # corruption this block catches, and stops short of the elaborated one: the
    # index budget allows at most (K - SCALE_D) * (LEN - 1) + SCALE_D = 1531.
    assert peak[0] > narrower_max, \
        f'the unipolar scaled accumulator peaked at {peak[0]}, inside a {ACC_WIDTH - 1}-bit clamp'
    assert peak[0] < acc_max, \
        f'the unipolar scaled accumulator reached {peak[0]}, at the elaborated clamp {acc_max}'
    # A unipolar accumulator has no offset and its carry only subtracts down to
    # zero, so its negative rail is unreachable by construction, not for want of
    # stimulus. The bipolar arm moves at 3.5 per cycle in both directions, which
    # neither clamp is within reach of at this ENTRY and SEQ_WIDTH.
    assert bottom[0] == 0, f'the unipolar scaled accumulator went negative, to {bottom[0]}'
    assert narrower_max > peak[1] > 0, f'the bipolar scaled accumulator peaked at {peak[1]}'
    assert narrower_min < bottom[1] < 0, f'the bipolar scaled accumulator bottomed at {bottom[1]}'
    assert acc_min < narrower_min, 'the narrower clamp is not inside the elaborated one'
    # The inequalities above bound the reaches; these pin them to the figures the
    # documentation quotes.
    reached = (peak[0], peak[1], bottom[1])
    recorded = (SAT_PEAK_UNI, SAT_PEAK_BI, SAT_TROUGH_BI)
    assert reached == recorded, \
        f'geometry d reached {reached}, but the recorded excursions are {recorded}'
    return peak[0], bottom[1]


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
        emit_row(rows, arms, index, timestep == 0)
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
    one table serves the unipolar and the bipolar DUTs. Geometry d's railed codes
    go to a second table in the same layout.
    """
    for suffix, weights, biases in [("", WEIGHT_CODES, BIAS_CODES),
                                    ("_d", WEIGHT_CODES_D, BIAS_CODES_D)]:
        codes = list(weights) + list(biases)
        (VEC_DIR / f"conv_ugemm_operand{suffix}.hex").write_text(
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
    peak, depth = run_saturation(rows)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:1 + 3 * TIMESTEPS], \
        'reset() did not restore the opening sequence-index, counter and accumulator state'
    # A shared input code pair would put the same probability grid through both
    # encoders, leaving the bipolar arm no distinct stimulus.
    columns = [row.split() for row in rows[1:1 + 3 * TIMESTEPS]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        'bipolar input stimulus is identical to unipolar'

    write_rom(layer)
    write_operands()
    pp_delay = layer.hw.pp_delay
    defines = [("BATCH", BATCH), ("IN_CHANNELS", IN_CHANNELS), ("IN_H", IN_H), ("IN_W", IN_W),
               ("OUT_CHANNELS", OUT_CHANNELS), ("KERNEL_H", KERNEL[0]), ("KERNEL_W", KERNEL[1]),
               ("SEQ_WIDTH", SEQ_WIDTH), ("WIDTH", ACC_WIDTH),
               ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
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
        + f", excursion {peak} / {depth})"
    )


if __name__ == "__main__":
    main()
