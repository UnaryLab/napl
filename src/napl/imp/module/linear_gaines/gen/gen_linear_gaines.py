"""
Generate golden test vectors for the linear_gaines module RTL straight from napl's
functional Python models (napl.sim.module.linear_gaines1 and linear_gaines2) -- so
the testbench checks the Verilog against the *actual* simulator, not a hand-derived
truth table.

One RTL module per polarity serves both classes. Under a matched configuration the
two build the same weight-threshold table, the same bias sequence and the same
scaled-adder threshold sequence, so they differ only in the ROM image, in state the
class never reads (linear_gaines2's scale_seq and its unipolar non-scaled counter),
and in their Python surface. Every arm below therefore holds one instance of each
class and requires their outputs to agree bit for bit on every timestep; the golden
column is that common output.

linear_gaines reduces one lane per output feature. Its per-timestep parallel count is

    unipolar   pc = sum_f x_f * w_f                       = popcount(AND(x, w))
    bipolar    pc = 2*sum_f x_f*w_f - sum_f x_f
                    - sum_f w_f + IN_FEATURES             = popcount(XNOR(x, w))

with the bias spike as one more addend, so the RTL lane is IN_FEATURES weight
comparators feeding IN_FEATURES mul_gaines_<polarity> cells plus one encode cell for
the bias. SCALED then selects the Gaines adder at elaboration:

    scaled      encode over the threshold sequence  -- the model's reference_encode
    non-scaled  add_gaines at SCALED = 0 (unipolar) -- pc > 0
                a DEPTH-bit saturating counter (bipolar)

Weight and bias are held fixed-point codes rather than spike streams (both classes
hold one threshold sequence per input feature, which is what internal_encode = True
records), so they go to ../vec/linear_gaines_operand.hex instead of vector columns.
The input spikes are captured from the model's own encoder with a forward hook, so
that column is provably the bits the model reduced. The weight thresholds have no
encoder object to hook -- the classes inline the comparison -- so their ROM is read
from the model's own w_num_seq buffer, and the bias and scaled-adder ROMs from the
model's own b_encoder.num_seq and reference_encode.num_seq.

Every operand and every input value sits on the 1/W_LEN probability grid, so the
model's float comparison and the RTL's fixed-point comparison are the same compare.

Six DUT configurations share each row, one column of expected output each: three
arms per polarity.

  a  -- scaled, with bias    (entry = IN_FEATURES + 1)
  b  -- scaled, no bias      (entry = IN_FEATURES)
  c  -- non-scaled, with bias (OR reduction in unipolar, counter in bipolar)

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so the
two polarities run as parallel columns rather than sequential blocks: each row drives
both polarity DUTs with their own encoded input stream. Three sequences follow each
other: one over the first input vector, one over the second, then the first replayed
after the second has dirtied the state and reset() has cleared it, which must
reproduce the first sequence bit for bit. The dirty block is DIRTY timesteps long,
which is neither a multiple of the threshold period nor of the scaled-adder period,
and the generator requires the bipolar counter to sit away from its reset value
cnt_half when the block ends -- so a reset that misses the counter, either sequence
index or an encoder index shows up in the replayed block.

A fourth block rails the operands, which is what makes DEPTH observable. DEPTH bounds
a counter, and away from its clamps the trajectory relative to cnt_half is the same
whatever DEPTH is, so only a clamped excursion changes a compared output. Lane 0
carries railed weight and bias codes for that purpose. Its weight codes are full
scale, so an all-high input block spikes every addend and the counter climbs by ENTRY
per cycle onto CNT_MAX; its bias code is full scale too, so an all-low input block
still spikes that one addend and the counter falls by ENTRY - 2 per cycle onto 0.
The ENTRY - 2 rate is the railed-high bias: a lane with no bias addend, or one whose
bias is not railed, falls by ENTRY per cycle instead.

What discriminates DEPTH is a departure from a clamp, not the clamp itself. The
opening charge block contributes no mismatch, because a counter of any DEPTH sits on
its own high clamp there and emits. The drain block is where the column parts, since
a narrower counter starts its descent closer to its own cnt_half, and it is
sufficient on its own: with the third block dropped, GEN_DEPTH 8 -> 7 still fails, at
n=807 with 4 mismatches over 836 vectors. The third block is kept for the low clamp
rather than for DEPTH: it is the only stimulus that leaves the low rail, so deleting
the low clamp from the RTL fails at n=844 with the block present and PASSes 836/836
with it dropped.

Output: ../vec/linear_gaines.vec, one line per timestep:

    <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c> <out_b_a> <out_b_b> <out_b_c>

`in_*` is IN_FEATURES binary digits, MSB first, so input feature f occupies
i_input_spike[f]. Each `out_*` column is LANES digits, lane l the output feature l.

Sizing values come from this file only and are emitted into
../vec/linear_gaines_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay and the row
count, which the testbench compares against the rows it consumed.

arm.check_mapping() translates this module's own mapping.yaml entries -- both classes,
every elaborated arm -- and requires the resolved parameters to equal the ones used
here, so the co-simulation gates the mapping entries as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_linear_gaines.py
"""
import math
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import linear_gaines1, linear_gaines2
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "linear_gaines.vec"
PARAMS = VEC_DIR / "linear_gaines_params.vh"

# Shapes and codec mirror tests/module/test_linear_gaines1.py's fidelity-scale input.
IN_FEATURES = 16
LANES = 8
TIMESTEP = 256
DEPTH = 8

SEQ_WIDTH = math.ceil(math.log2(TIMESTEP))
W_LEN = 2 ** SEQ_WIDTH
TIMESTEPS = W_LEN
# Neither a multiple of W_LEN nor of the scaled-adder period, so no index aliases
# back to its reset value; the counter is checked directly.
DIRTY = TIMESTEPS // 2 + 1
# Lane 0 is railed, so its parallel count is ENTRY on an all-high input and 1 (the
# railed bias alone) on an all-low one, and the counter integrates 2*count - ENTRY:
# it climbs by ENTRY and falls by ENTRY - 2, the slower of the two.
ENTRY = IN_FEATURES + 1
CNT_MAX = 2 ** DEPTH - 1
# A rail block must carry the counter across that whole span at the slower rate and
# then hold it on the clamp while the next block's departure is read against it, so
# the block is twice the span.
RAIL_STEPS = 2 * -(-CNT_MAX // (ENTRY - 2))

# label, scaled, has_bias
ARMS = [
    ("a", True, True),
    ("b", True, False),
    ("c", False, True),
]


def operand_codes(count, stride, offset):
    """Return `count` fixed-point codes in [0, W_LEN] on the 1/W_LEN grid."""
    return [(i * stride + offset) % (W_LEN + 1) for i in range(count)]


# Lane 0 is railed high: its weight and bias codes are the full-scale code, so its
# spikes are 1 whatever the timestep. That lane is what an input rail drives onto the
# counter clamps, and it needs no second operand table to do it.
WEIGHT_CODES = [W_LEN] * IN_FEATURES + operand_codes((LANES - 1) * IN_FEATURES, 29, 5)
BIAS_CODES = [W_LEN] + operand_codes(LANES - 1, 53, 7)
# One code set per polarity: a shared set would put the same probability grid through
# both encoders, so the bipolar input column would copy the unipolar one. The weight
# and bias codes stay shared, since one operand file feeds every DUT.
INPUT_CODES = {
    'unipolar': [operand_codes(IN_FEATURES, 71, 13), operand_codes(IN_FEATURES, 37, 101)],
    'bipolar': [operand_codes(IN_FEATURES, 43, 97), operand_codes(IN_FEATURES, 59, 151)],
}
# The rail vectors: every feature at full scale, then every feature at zero.
RAIL_CODES = [[W_LEN] * IN_FEATURES, [0] * IN_FEATURES]


def to_value(code, polarity):
    """Map a fixed-point probability code to the polarity's real value."""
    prob = code / W_LEN
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
    """One polarity's input encoder plus its three arms, each in both classes."""


    def __init__(self, polarity):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        require_seeded_sys(codec)
        self.weight = tensor(WEIGHT_CODES, (LANES, IN_FEATURES), polarity)
        self.bias = tensor(BIAS_CODES, (LANES,), polarity)
        self.values = [tensor(codes, (IN_FEATURES,), polarity)
                       for codes in INPUT_CODES[polarity] + RAIL_CODES]
        self.enc = encode(codec)
        # The input spikes the layers reduce are captured from the encoder's own
        # forward, so the golden column is provably the bits the model saw.
        self.input_spike = capture(self.enc)
        self.configs = []
        self.pairs = []
        for _, scaled, has_bias in ARMS:
            config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                      'dim': 2, 'scaled': scaled, 'depth': DEPTH}
            require_seeded_sys(config)
            self.configs.append(config)
            bias = self.bias if has_bias else None
            self.pairs.append((linear_gaines1(self.weight, bias, config),
                               linear_gaines2(self.weight, bias, dict(config, seed=1))))
        # The width the scaled arm's threshold ROM is emitted at, which is what
        # check_mapping requires the mapping's SCALE_WIDTH expression to resolve to.
        self.scale_width = self.pairs[0][0].scale_width
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python models, so without this
        the mapping entries could drift from the hardware the co-simulation verifies.
        Every elaborated arm is translated, in both classes: they differ in `scaled`
        and in the bias, which change SCALED, HAS_BIAS and the entry SCALE_WIDTH is
        derived from.

        SCALE_WIDTH is checked against the model's own `scale_width`, the width that
        sizes the emitted threshold ROM, rather than against a second copy of the
        mapping's rounding: comparing the expression to a recomputation of itself
        would pass whatever the model built. Only a scaled arm carries the attribute,
        and every arm elaborates the same value, which SCALED = 0 leaves unused.
        """
        for (_, scaled, has_bias), config in zip(ARMS, self.configs):
            expected = {"IN_FEATURES": IN_FEATURES, "LANES": LANES,
                        "SEQ_WIDTH": SEQ_WIDTH,
                        "SCALE_WIDTH": self.scale_width,
                        "HAS_BIAS": int(has_bias), "SCALED": int(scaled)}
            if self.polarity == 'bipolar':
                expected["DEPTH"] = DEPTH
            for class_name in ("linear_gaines1", "linear_gaines2"):
                node = {"class": class_name,
                        "config": {"weight": self.weight,
                                   "bias": self.bias if has_bias else None,
                                   "config": config, "lanes": LANES}}
                binding = translate_node(node)
                assert binding.rtl_module == f"linear_gaines_{self.polarity}", binding.rtl_module
                assert binding.parameters == expected, \
                    f"mapping.yaml {class_name} resolves {binding.parameters}"


    def reset(self):
        """Restart the input encoder and every layer of both classes."""
        self.enc.reset()
        for pair in self.pairs:
            for layer in pair:
                layer.reset()


    def step(self, index):
        """Advance one timestep over input vector `index`; return the row columns."""
        spike = self.enc(self.values[index])
        outputs = []
        for (first, second), (label, _, _) in zip(self.pairs, ARMS):
            out = bits_of(first(spike))
            assert out == bits_of(second(spike)), \
                f'linear_gaines1 and linear_gaines2 disagree on the {self.polarity} {label} arm'
            outputs.append(out)
        return bits_of(self.input_spike.spike), outputs


    def counter(self):
        """The non-scaled arm's counter, or None when this polarity has none."""
        layer = self.pairs[-1][0]
        return None if layer.polarity == 'unipolar' else layer.cnt


def emit_row(rows, arms, index, first):
    """Step both arms over input vector `index` and append their golden row."""
    inputs = []
    outputs = []
    for one in arms:
        in_bits, out_bits = one.step(index)
        inputs.append(as_binary(in_bits))
        outputs.append([as_binary(bits) for bits in out_bits])
    columns = [f"{1 if first else 0}"] + inputs + outputs[0] + outputs[1]
    rows.append(" ".join(columns))


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps over the other input vector run before the recorded sequence
    and are followed by reset(), so the recorded block starts from sequence indices,
    encoder indices and a counter that held state. The counter is required to sit
    away from its reset value when the dirty block ends, so an unreset counter
    cannot reproduce the recorded block by chance. Those outputs must equal the same
    block recorded from fresh models, which the testbench checks by pulsing i_rst_n
    on the opening rst=1 row.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    for one in arms:
        one.check_mapping()
    if dirty:
        for _ in range(dirty):
            for one in arms:
                one.step(1 - index)
        counter = arms[1].counter()
        half = arms[1].pairs[-1][0].cnt_half
        assert not torch.any(counter == half), \
            f'the dirty block left a bipolar counter at its reset value {half}'
        for one in arms:
            one.reset()

    for timestep in range(TIMESTEPS):
        emit_row(rows, arms, index, timestep == 0)
    return arms


def run_rails(rows):
    """Append the railed block, which is what makes DEPTH observable.

    Lane 0 holds railed weight and bias codes, so an all-high input block takes its
    parallel count to ENTRY and its bipolar counter climbs by ENTRY per cycle onto
    CNT_MAX; an all-low input block leaves only the railed bias addend, so the counter
    falls by ENTRY - 2 per cycle onto 0. The block charges up, drains onto the low
    rail, and charges back. The drain is what makes DEPTH observable: a narrower
    counter leaves its own high clamp closer to its own cnt_half and crosses it
    sooner, which moves the compared column. The charge-back block is what makes the
    low clamp observable, being the only stimulus that leaves the low rail.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    # Index 2 is the all-high input vector, index 3 the all-low one.
    schedule = [2] * RAIL_STEPS + [3] * RAIL_STEPS + [2] * RAIL_STEPS
    peak = 0
    bottom = 2 ** DEPTH
    for step, index in enumerate(schedule):
        emit_row(rows, arms, index, step == 0)
        lane0 = int(arms[1].counter().reshape(-1)[0])
        peak = max(peak, lane0)
        bottom = min(bottom, lane0)
    layer = arms[1].pairs[-1][0]
    assert peak == layer.cnt_max, \
        f'the railed bipolar counter peaked at {peak}, short of {layer.cnt_max}'
    assert bottom == 0, f'the railed bipolar counter bottomed at {bottom}, short of 0'


def sequence_rows(sequence, length, name):
    """Render a model number sequence as `length`-bit binary ROM lines."""
    values = sequence.detach().float().reshape(-1)
    count = values.numel()
    lines = []
    for index in range(count):
        scaled = values[index].item() * count
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f'{name}[{index}] is off the 1/{count} grid'
        lines.append(f"{code:0{length}b}")
    return lines


def write_roms(arms):
    """Emit the three ROM images, read from the models' own buffers.

    The weight thresholds have no encoder object to hook, so they come from the
    model's w_num_seq buffer; the bias and scaled-adder sequences come from the
    b_encoder and reference_encode the model built. Every arm and both classes are
    required to hold the same tables, which is what lets one RTL module per polarity
    serve both classes.
    """
    reference = arms[0].pairs[0][0]
    for one in arms:
        for pair, (label, scaled, has_bias) in zip(one.pairs, ARMS):
            for layer in pair:
                assert torch.equal(layer.w_num_seq, reference.w_num_seq), \
                    f'the {one.polarity} {label} arm holds a different weight table'
                if has_bias:
                    assert torch.equal(layer.b_encoder.num_seq,
                                       reference.b_encoder.num_seq), \
                        f'the {one.polarity} {label} arm holds a different bias sequence'
                if scaled:
                    assert torch.equal(layer.reference_encode.num_seq,
                                       reference.reference_encode.num_seq), \
                        f'the {one.polarity} {label} arm holds a different threshold sequence'

    thresholds = reference.w_num_seq.detach().float()
    assert tuple(thresholds.shape) == (W_LEN, IN_FEATURES), thresholds.shape
    rows = []
    for timestep in range(W_LEN):
        codes = []
        # Feature f occupies bits [f*SEQ_WIDTH +: SEQ_WIDTH], so the highest feature
        # index is written first.
        for feature in reversed(range(IN_FEATURES)):
            scaled = thresholds[timestep, feature].item() * W_LEN
            code = round(scaled)
            assert abs(scaled - code) < 1e-9, \
                f'w_num_seq[{timestep}, {feature}] is off the 1/{W_LEN} grid'
            codes.append(f"{code:0{SEQ_WIDTH}b}")
        rows.append("".join(codes))
    (VEC_DIR / "linear_gaines_w.hex").write_text("\n".join(rows) + "\n")

    (VEC_DIR / "encode_rom.hex").write_text(
        "\n".join(sequence_rows(reference.b_encoder.num_seq, SEQ_WIDTH, 'b_encoder')) + "\n")
    scale_width = reference.scale_width
    (VEC_DIR / "gaines_rom.hex").write_text(
        "\n".join(sequence_rows(reference.reference_encode.num_seq, scale_width,
                                'reference_encode')) + "\n")
    return scale_width


def write_operands():
    """Emit the held operand codes: LANES*IN_FEATURES weights, then LANES biases.

    An operand code is the probability code both polarities compare against, so one
    table serves the unipolar and the bipolar DUTs.
    """
    codes = list(WEIGHT_CODES) + list(BIAS_CODES)
    (VEC_DIR / "linear_gaines_operand.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in codes) + "\n"
    )


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    header = ["rst", "in_u", "in_b"]
    header += [f"out_{polarity}_{label}" for polarity in "ub" for label, _, _ in ARMS]
    rows = [" ".join(header)]
    arms = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=DIRTY)
    rail_start = len(rows)
    run_rails(rows)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:1 + 3 * TIMESTEPS], \
        'reset() did not restore the opening index and counter state'
    # A railed block only makes DEPTH observable if the clamped column moves, which
    # is what the recovery on either side of each clamp does. Lane 0 is the railed
    # lane and the rightmost character of the column.
    block = [row.split()[-1][-1] for row in rows[rail_start:]]
    assert len(set(block)) > 1, 'the railed block never moved the bipolar counter column'
    # A shared input code set would put the same probability grid through both
    # encoders, leaving the bipolar arm no distinct stimulus.
    columns = [row.split() for row in rows[1:]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        'bipolar input stimulus is identical to unipolar'

    scale_width = write_roms(arms)
    write_operands()
    pp_delay = arms[0].pairs[0][0].hw.pp_delay
    defines = [("IN_FEATURES", IN_FEATURES), ("LANES", LANES), ("SEQ_WIDTH", SEQ_WIDTH),
               ("SCALE_WIDTH", scale_width), ("DEPTH", DEPTH),
               ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    for label, scaled, has_bias in ARMS:
        suffix = label.upper()
        defines += [(f"SCALED_{suffix}", int(scaled)), (f"HAS_BIAS_{suffix}", int(has_bias))]
    PARAMS.write_text("".join(f"`define GEN_{name} {value}\n" for name, value in defines))
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} ("
        + ", ".join(f"GEN_{name}={value}" for name, value in defines)
        + ")"
    )


if __name__ == "__main__":
    main()
