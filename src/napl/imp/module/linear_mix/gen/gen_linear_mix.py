"""
Generate golden test vectors for the linear_mix module RTL straight from napl's
functional Python model (napl.sim.module.linear_mix) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

linear_mix reduces one lane per output feature. Its per-timestep partial sum is a
popcount of Gaines products:

    unipolar   psum = sum_f x_f * w_f                     = popcount(AND(x, w))
    bipolar    psum = 2*sum_f x_f*w_f - sum_f x_f
                      - sum_f w_f + IN_FEATURES
                    = sum_f (1 - x_f - w_f + 2*x_f*w_f)   = popcount(XNOR(x, w))

with the bias spike added as one more addend. The weight and bias encoders are
held inside the layer, so the RTL re-encodes both from held fixed-point codes on
its ports every timestep. The RTL lane is therefore IN_FEATURES
(encode -> mul_gaines_<polarity>) cells plus the optional (encode -> bias) addend
feeding one add_scale_<polarity>:

    encode                -- one per (output feature, input feature) weight bit
    mul_gaines_<polarity> -- the Gaines product of the input spike and weight spike
    encode                -- one per output feature, the free-running bias bit
    add_scale_<polarity>    -- one per output feature, the scaled accumulator

The weight and the bias sequences are distinct Sobol dimensions (weight on dim,
bias on dim + 1), so this file writes two ROMs: vec/lm_wrom.hex from the weight
encoder and vec/lm_brom.hex from the bias encoder, both relative to the simulation
cwd (module/linear_mix/). The RTL encode cells $readmemb those tables.

Six DUT configurations share each row, one column of expected output each:
  out_u     -- unipolar, with bias   (entry = scale = in_features + 1)
  out_u_nb  -- unipolar, no bias     (entry = scale = in_features)
  out_u_s   -- unipolar, with bias, scale SCALE_S != entry, railed operands
  out_b     -- bipolar,  with bias
  out_b_nb  -- bipolar,  no bias
  out_b_s   -- bipolar,  with bias, scale SCALE_S != entry, railed operands

The two SCALE_S arms are the only accumulators that move: with scale == entry
the bipolar offset (entry - scale) / 2 is 0 and a carry subtracts the whole
entry, so the accumulator stays inside [0, entry - 1]. They carry the positive
and the negative saturation blocks, which is what makes WIDTH and the negative
clamp observable.

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded input stream. Two sequences
follow each other: a fresh stream, then the same stream replayed after half a
stream has dirtied the sequence indices and the accumulators and reset() has
cleared them, which must reproduce the first sequence bit for bit.

Output: ../vec/linear_mix.vec, one line per timestep:

    <rst> <in_u_bits> <in_b_bits> <out_u> <out_u_nb> <out_u_s> <out_b> <out_b_nb>
    <out_b_s>

`in_*_bits` is IN_FEATURES binary digits, MSB first, so input feature f occupies
i_input[f]. `out_*` is LANES binary digits, MSB first, with lane l the
output feature l.

Weight and bias operands are held fixed-point codes, so they go to
../vec/linear_mix_operand_u.hex and _b.hex (LANES*(IN_FEATURES+1) lines each:
IN_FEATURES weight codes then the bias code, per lane) instead of a vector column.
The scaled arms' railed operands go to ../vec/linear_mix_operand_s.hex in the
same layout; an operand code is the probability code either polarity compares
against, so one file serves both.

Sizing values come from this file only and are emitted into
../vec/linear_mix_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_linear_mix.py
"""
import math
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import linear_mix
from napl.sim.operation import encode
from napl.syn import translate_node

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "linear_mix.vec"
PARAMS = VEC_DIR / "linear_mix_params.vh"

# Shapes and codec mirror tests/module/test_linear_mix.py.
IN_FEATURES = 16
LANES = 8
TIMESTEP = 256
ACC_WIDTH = 12

SEQ_WIDTH = math.ceil(math.log2(TIMESTEP))
LEN = 2 ** SEQ_WIDTH
TIMESTEPS = TIMESTEP


def operand_codes(count, stride, offset):
    """Return `count` fixed-point codes in [0, LEN] on the 1/LEN operand grid."""
    return [(i * stride + offset) % (LEN + 1) for i in range(count)]


WEIGHT_CODES = operand_codes(LANES * IN_FEATURES, 29, 5)
BIAS_CODES = operand_codes(LANES, 53, 7)
# One code list per polarity: a shared list would put the same probability grid
# through both encoders, so the bipolar input column would copy the unipolar one.
INPUT_CODES = {'unipolar': operand_codes(IN_FEATURES, 71, 13),
               'bipolar': operand_codes(IN_FEATURES, 43, 97)}

# A divisor below the fan-in, so the bipolar offset (entry - scale) / 2 is nonzero and the lane accumulator can drift.
SCALE_S = 5
RAIL_WEIGHT = LEN
RAIL_BIAS = 0
# Charge for LEN - 1 cycles then drain for LEN - 1, enough to drive both scaled accumulators onto the positive clamp and back.
SAT_CHARGE = LEN - 1
SAT_DRAIN = LEN - 1
# Drain for LEN - 1 cycles then recharge for 200, enough to drive the bipolar accumulator past the next narrower clamp and back.
NEG_DRAIN = LEN - 1
NEG_CHARGE = 200
# Rails as model values: probability 1 and 0 in the polarity's own domain.
RAIL_VALUE = {'unipolar': (1.0, 0.0), 'bipolar': (1.0, -1.0)}


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


class arm:
    """One polarity's encoder plus its with-bias and no-bias layer models."""


    def __init__(self, polarity):
        self.polarity = polarity
        # Input on dim 1, weight on dim 2, bias on dim 3: three distinct Sobol
        # dimensions decorrelate the streams.
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'dim': 2, 'scale': None, 'width': ACC_WIDTH}
        weight = tensor(WEIGHT_CODES, (LANES, IN_FEATURES), polarity)
        bias = tensor(BIAS_CODES, (LANES,), polarity)
        self.value = tensor(INPUT_CODES[polarity], (IN_FEATURES,), polarity)
        self.config_s = dict(config, scale=SCALE_S)
        rail_weight = tensor([RAIL_WEIGHT] * (LANES * IN_FEATURES),
                             (LANES, IN_FEATURES), polarity)
        rail_bias = tensor([RAIL_BIAS] * LANES, (LANES,), polarity)
        self.enc = encode(codec)
        self.layer = linear_mix(weight, bias, config)
        self.layer_nb = linear_mix(weight, None, config)
        self.layer_s = linear_mix(rail_weight, rail_bias, self.config_s)
        # A one-bit-narrower shadow of the scaled arm, never emitted but required to part from it so the elaborated WIDTH is observable.
        self.layer_narrow = linear_mix(rail_weight, rail_bias,
                                       dict(self.config_s, width=ACC_WIDTH - 1))
        self.rail_weight = rail_weight
        self.rail_bias = rail_bias
        self.config = config
        self.weight = weight
        self.bias = bias
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Both elaborated configurations are translated: with bias and
        without, which change ENTRY and so the default scale, plus the explicit
        scale, which takes the other branch of the SCALE expression.
        """
        node = {"class": "linear_mix",
                "config": {"weight": self.weight, "bias": self.bias,
                           "config": self.config, "lanes": LANES}}
        expected = {"IN_FEATURES": IN_FEATURES, "LANES": LANES, "SEQ_WIDTH": SEQ_WIDTH,
                    "WIDTH": ACC_WIDTH, "HAS_BIAS": 1, "SCALE": IN_FEATURES + 1}
        binding = translate_node(node)
        assert binding.rtl_module == f"linear_mix_{self.polarity}", binding.rtl_module
        assert binding.parameters == expected, \
            f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"

        node_nb = dict(node, config=dict(node["config"], bias=None))
        expected_nb = dict(expected, HAS_BIAS=0, SCALE=IN_FEATURES)
        binding_nb = translate_node(node_nb)
        assert binding_nb.parameters == expected_nb, \
            f"mapping.yaml {binding_nb.rtl_module} resolves {binding_nb.parameters}"

        node_s = dict(node, config=dict(node["config"], config=self.config_s))
        expected_s = dict(expected, SCALE=SCALE_S)
        binding_s = translate_node(node_s)
        assert binding_s.parameters == expected_s, \
            f"mapping.yaml {binding_s.rtl_module} resolves {binding_s.parameters}"


    def reset(self):
        """Restart the encoder and every layer."""
        self.enc.reset()
        self.layer.reset()
        self.layer_nb.reset()
        self.layer_s.reset()
        self.layer_narrow.reset()


    def step(self, value=None):
        """Advance one timestep, over `value` when given; return the row columns."""
        spike = self.enc(self.value if value is None else value)
        out = self.layer(spike)
        out_nb = self.layer_nb(spike)
        out_s = self.layer_s(spike)
        self.narrow_bits = bits_of(self.layer_narrow(spike))
        return bits_of(spike), bits_of(out), bits_of(out_nb), bits_of(out_s)


def emit_row(rows, arms, rail, first):
    """Step both arms, at the given rail when driven, and append their golden row.

    Returns whether either arm's scaled output parted from its one-bit-narrower
    shadow on this cycle, which is the row a corrupted WIDTH would fail on.
    """
    columns = [f"{1 if first else 0}"]
    parted = False
    outputs = []
    for one in arms:
        value = None
        if rail is not None:
            level = RAIL_VALUE[one.polarity][0 if rail else 1]
            value = torch.full((IN_FEATURES,), level, dtype=global_config.ntype)
        in_bits, out_bits, out_nb_bits, out_s_bits = one.step(value)
        parted = parted or out_s_bits != one.narrow_bits
        columns.append(as_binary(in_bits))
        outputs.append(as_binary(out_bits))
        outputs.append(as_binary(out_nb_bits))
        outputs.append(as_binary(out_s_bits))
    rows.append(" ".join(columns + outputs))
    return parted


def run_sequence(rows, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps run before the recorded sequence and are followed by
    reset(), so the recorded block starts from sequence indices and accumulators
    that held state. Its outputs must equal the same block recorded from fresh
    models, which the testbench checks by pulsing i_rst_n on the opening rst=1 row.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    for one in arms:
        one.check_mapping()
    if dirty:
        for _ in range(dirty):
            for one in arms:
                one.step()
        for one in arms:
            one.reset()

    for timestep in range(TIMESTEPS):
        emit_row(rows, arms, None, timestep == 0)
    return arms[0].layer


def run_saturation(rows):
    """Append the positive-clamp block, which is what makes WIDTH observable.

    The scaled arms hold their weights at the top code and their bias at 0, so a
    spiking input gives every lane a partial sum of IN_FEATURES and a silent one
    gives 0. With SCALE_S below ENTRY the spiking block charges both scaled
    accumulators onto their positive clamp at 2**(WIDTH-1) - 1 and the silent
    block drains them: the stored charge is how many further cycles an arm keeps
    emitting, so a narrower WIDTH stops emitting sooner and its column differs.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    rails = [True] * SAT_CHARGE + [False] * SAT_DRAIN
    peak = [0, 0]
    parted = False
    for step, rail in enumerate(rails):
        parted = emit_row(rows, arms, rail, step == 0) or parted
        for slot, one in enumerate(arms):
            peak[slot] = max(peak[slot], int(one.layer_s.acc.accumulator.reshape(-1).max()))
    assert parted, 'the positive block never parted from a one-bit-narrower accumulator'
    # The post-clamp carry retains the clamped unipolar accumulator at acc_max - SCALE_S, while the bipolar arm, climbing only at the offset, need only pass the next narrower clamp.
    acc_max = arms[0].layer_s.acc.acc_max
    assert peak[0] == acc_max - SCALE_S, \
        f'the unipolar scaled accumulator peaked at {peak[0]}, short of {acc_max}'
    assert peak[1] > acc_max // 2, \
        f'the bipolar scaled accumulator peaked at {peak[1]}, below {acc_max // 2}'
    return peak[1]


def run_negative(rows):
    """Append the negative-clamp block, which is what makes ACC_LO observable.

    Only a bipolar accumulator moves down, by the offset (ENTRY - SCALE_S) / 2
    that a scale below the fan-in creates, and only while its partial sum is 0,
    which is what the silent input block gives. The spiking block then recharges
    it: how long the arm stays silent on the way back up is what the clamp sets,
    so a shallower ACC_LO starts emitting sooner and its column differs.

    The unipolar accumulator has no offset and its carry only ever subtracts down
    to zero, so its negative clamp is unreachable by construction.
    """
    arms = [arm('unipolar'), arm('bipolar')]
    rails = [False] * NEG_DRAIN + [True] * NEG_CHARGE
    bottom = [0, 0]
    parted = False
    for step, rail in enumerate(rails):
        parted = emit_row(rows, arms, rail, step == 0) or parted
        for slot, one in enumerate(arms):
            bottom[slot] = min(bottom[slot], int(one.layer_s.acc.accumulator.reshape(-1).min()))
    assert parted, 'the negative block never parted from a one-bit-narrower accumulator'
    acc_min = arms[1].layer_s.acc.acc_min
    assert bottom[1] <= acc_min // 2, \
        f'the bipolar scaled accumulator bottomed at {bottom[1]}, above {acc_min // 2}'
    assert bottom[0] == 0, \
        f'the unipolar scaled accumulator went negative, to {bottom[0]}'
    return bottom[1]


def write_rom(layer):
    """Emit the weight and bias number-sequence ROMs the RTL encode cells read.

    The weight encoder sits on one Sobol dimension and the bias encoder on the
    next, so the two tables differ. A ROM that drifted from the model would change
    the compared outputs.
    """
    for encoder, name in [(layer.w_encoder, "lm_wrom.hex"),
                          (layer.b_encoder, "lm_brom.hex")]:
        num_seq = encoder.num_seq.detach().float().reshape(-1)
        assert num_seq.numel() == LEN, f'num_seq length {num_seq.numel()} != LEN {LEN}'
        lines = []
        for index in range(LEN):
            scaled = num_seq[index].item() * LEN
            code = round(scaled)
            assert abs(scaled - code) < 1e-9, f'{name} num_seq[{index}] off the 1/{LEN} grid'
            lines.append(f"{code:0{SEQ_WIDTH}b}")
        (VEC_DIR / name).write_text("\n".join(lines) + "\n")


def write_operands():
    """Emit per-lane operand codes: IN_FEATURES weights then the bias, per lane."""
    for polarity, suffix in [('unipolar', 'u'), ('bipolar', 'b')]:
        lines = []
        for lane in range(LANES):
            for feature in range(IN_FEATURES):
                lines.append(WEIGHT_CODES[lane * IN_FEATURES + feature])
            lines.append(BIAS_CODES[lane])
        (VEC_DIR / f"linear_mix_operand_{suffix}.hex").write_text(
            "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in lines) + "\n"
        )
    # The scaled arms' railed operands, shared by both polarities since an operand code is the probability code each compares against.
    rail = []
    for _ in range(LANES):
        rail += [RAIL_WEIGHT] * IN_FEATURES + [RAIL_BIAS]
    (VEC_DIR / "linear_mix_operand_s.hex").write_text(
        "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in rail) + "\n"
    )


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["rst in_u_bits in_b_bits out_u out_u_nb out_u_s out_b out_b_nb out_b_s"]
    layer = run_sequence(rows)
    run_sequence(rows, dirty=TIMESTEPS // 2)
    assert rows[1:1 + TIMESTEPS] == rows[1 + TIMESTEPS:1 + 2 * TIMESTEPS], \
        'reset() did not restore the opening sequence-index and accumulator state'
    peak = run_saturation(rows)
    depth = run_negative(rows)
    # A shared input code list would put the same probability grid through both
    # encoders, leaving the bipolar arm no distinct stimulus.
    columns = [row.split() for row in rows[1:1 + 2 * TIMESTEPS]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        'bipolar input stimulus is identical to unipolar'

    write_rom(layer)
    write_operands()
    pp_delay = layer.hw.pp_delay
    PARAMS.write_text(
        f"`define GEN_IN_FEATURES {IN_FEATURES}\n"
        f"`define GEN_LANES {LANES}\n"
        f"`define GEN_SEQ_WIDTH {SEQ_WIDTH}\n"
        f"`define GEN_WIDTH {ACC_WIDTH}\n"
        f"`define GEN_SCALE {IN_FEATURES + 1}\n"
        f"`define GEN_SCALE_NB {IN_FEATURES}\n"
        f"`define GEN_SCALE_S {SCALE_S}\n"
        f"`define GEN_PP_DELAY {pp_delay}\n"
        f"`define GEN_VECTORS {len(rows) - 1}\n"
    )
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} "
        f"(GEN_IN_FEATURES={IN_FEATURES}, GEN_LANES={LANES}, "
        f"GEN_SEQ_WIDTH={SEQ_WIDTH}, GEN_WIDTH={ACC_WIDTH}, "
        f"GEN_SCALE={IN_FEATURES + 1}, GEN_SCALE_NB={IN_FEATURES}, "
        f"GEN_SCALE_S={SCALE_S}, excursion {peak} / {depth}, "
        f"GEN_PP_DELAY={pp_delay})"
    )


if __name__ == "__main__":
    main()
