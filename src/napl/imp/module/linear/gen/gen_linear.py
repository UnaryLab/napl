"""
Generate golden test vectors for the linear module RTL straight from napl's
functional Python model (napl.sim.module.linear) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

linear reduces one lane per output feature. Its per-timestep partial sum is a
popcount of Gaines products:

    unipolar   psum = sum_f x_f * w_f                     = popcount(AND(x, w))
    bipolar    psum = 2*sum_f x_f*w_f - sum_f x_f
                      - sum_f w_f + IN_FEATURES
                    = sum_f (1 - x_f - w_f + 2*x_f*w_f)   = popcount(XNOR(x, w))

with the bias spike added as one more addend. add_any popcounts its ENTRY-bit
input and takes the same ENTRY for its bipolar offset, so the RTL lane is
IN_FEATURES mul_gaines_<polarity> cells plus the optional bias bit feeding one
add_any_<polarity>:

    mul_gaines_<polarity>  -- one per (output feature, input feature)
    add_any_<polarity>     -- one per output feature, the scaled accumulator

The model re-encodes the weight and the bias from externally held tensors every
timestep, so both arrive on RTL ports rather than being rebuilt in hardware. The
spikes are captured from the model's own encoders with forward hooks, so the
columns are the exact bits the model reduced.

Six DUT configurations share each row, one column of expected output each:
  out_u     -- unipolar, with bias   (entry = scale = in_features + 1)
  out_u_nb  -- unipolar, no bias     (entry = scale = in_features)
  out_b     -- bipolar,  with bias
  out_b_nb  -- bipolar,  no bias
  out_u_s   -- unipolar, with bias, scale SCALE_S != entry
  out_b_s   -- bipolar,  with bias, scale SCALE_S != entry

The two SCALE_S arms are the only accumulators that move: with scale == entry
the bipolar offset (entry - scale) / 2 is 0 and a carry subtracts the whole
entry, so the accumulator stays inside [0, entry - 1]. They carry the positive
and the negative saturation blocks, which is what makes WIDTH and the negative
clamp observable.

The unipolar and the bipolar value grids span different ranges, so the encoded
bipolar rate p = (x + 1) / 2 is a different stimulus from the unipolar rate.

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded streams. Three sequences follow
each other: one over the first input vector, one over the second, then the first
replayed after the second has dirtied the accumulators and reset() has cleared
them, which must reproduce the first sequence bit for bit.

Output: ../vec/linear.vec, one line per timestep:

    <rst> <in_u> <in_b> <w_u> <w_b> <b_u> <b_b> <out_u> <out_u_nb> <out_b>
    <out_b_nb> <out_u_s> <out_b_s>

`in_*` is IN_FEATURES binary digits, MSB first, so input feature f occupies
i_input_spike[f]. `w_*` is LANES*IN_FEATURES digits with lane l, feature f at
i_weight[l*IN_FEATURES + f]. `b_*` and `out_*` are LANES digits, lane l the
output feature l.

Sizing values come from this file only and are emitted into
../vec/linear_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_linear.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import linear
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "linear.vec"
PARAMS = VEC_DIR / "linear_params.vh"

# Shapes and codec mirror tests/module/test_linear.py.
IN_FEATURES = 16
LANES = 8
TIMESTEP = 256
ACC_WIDTH = 12
TIMESTEPS = TIMESTEP
# A divisor other than the fan-in, so the fifth elaboration checks SCALE itself.
SCALE_S = 9
# Negative-clamp block: the bipolar scaled accumulator falls by the offset
# (ENTRY - SCALE_S) / 2 = 4 per cycle with the popcount at zero, so 2**(WIDTH-1)
# / 4 = 512 cycles reach the clamp; the charge back up runs at ENTRY - offset =
# 13 per cycle and needs 2**(WIDTH-1) / 13 = 158 to break silence.
NEG_DRAIN = 560
NEG_CHARGE = 200

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
    """One polarity's input encoder plus its with-bias and no-bias layer models."""


    def __init__(self, polarity, rail=None):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'dim': 2, 'scale': None, 'width': ACC_WIDTH}
        require_seeded_sys(codec, config)
        if rail is not None:
            # rail='pos' rails the bias with the weight, so the low input block
            # still carries the bias addend; rail='neg' rails it low, which is
            # what takes an interior lane's popcount to zero and lets the bipolar
            # offset drive the accumulator onto its negative clamp.
            low, high = RANGE[polarity][0], 1.0
            weight = torch.full((LANES, IN_FEATURES), high, dtype=global_config.ntype)
            bias = torch.full((LANES,), high if rail == 'pos' else low,
                              dtype=global_config.ntype)
            self.values = [torch.full((IN_FEATURES,), value, dtype=global_config.ntype)
                           for value in (low, high)]
        else:
            weight = grid(LANES * IN_FEATURES, (LANES, IN_FEATURES), polarity, 3)
            bias = grid(LANES, (LANES,), polarity, 1)
            self.values = [grid(IN_FEATURES, (IN_FEATURES,), polarity, offset)
                           for offset in (0, 5)]
        self.config_s = dict(config, scale=SCALE_S)
        require_seeded_sys(self.config_s)
        self.enc = encode(codec)
        self.layer = linear(weight, bias, config)
        self.layer_nb = linear(weight, None, config)
        self.layer_s = linear(weight, bias, self.config_s)
        self.weight_spike = capture(self.layer.w_encoder)
        self.weight_spike_nb = capture(self.layer_nb.w_encoder)
        self.bias_spike = capture(self.layer.b_encoder)
        self.config = config
        self.weight = weight
        self.bias = bias
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Every elaborated configuration is translated: with bias and
        without, which change ENTRY and so the default scale, plus the explicit
        scale, which takes the other branch of the SCALE expression.
        """
        node = {"class": "linear",
                "config": {"weight": self.weight, "bias": self.bias,
                           "config": self.config, "lanes": LANES}}
        expected = {"IN_FEATURES": IN_FEATURES, "LANES": LANES, "WIDTH": ACC_WIDTH,
                    "HAS_BIAS": 1, "SCALE": IN_FEATURES + 1}
        binding = translate_node(node)
        assert binding.rtl_module == f"linear_{self.polarity}", binding.rtl_module
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
        """Restart the input encoder and every layer."""
        self.enc.reset()
        self.layer.reset()
        self.layer_nb.reset()
        self.layer_s.reset()


    def step(self, index):
        """Advance one timestep over input vector `index`; return the row columns."""
        spike = self.enc(self.values[index])
        out = self.layer(spike)
        out_nb = self.layer_nb(spike)
        out_s = self.layer_s(spike)
        weight_bits = bits_of(self.weight_spike.spike)
        assert weight_bits == bits_of(self.weight_spike_nb.spike), \
            'the two layers encoded different weight spikes'
        return (bits_of(spike), weight_bits, bits_of(self.bias_spike.spike),
                bits_of(out), bits_of(out_nb), bits_of(out_s))


def emit_row(rows, arms, index, first):
    """Step both arms over input vector `index` and append their golden row."""
    inputs = []
    weights = []
    biases = []
    outputs = []
    scaled = []
    for one in arms:
        in_bits, weight_bits, bias_bits, out_bits, out_nb_bits, out_s_bits = one.step(index)
        inputs.append(as_binary(in_bits))
        weights.append(as_binary(weight_bits))
        biases.append(as_binary(bias_bits))
        outputs.append(as_binary(out_bits))
        outputs.append(as_binary(out_nb_bits))
        scaled.append(as_binary(out_s_bits))
    columns = [f"{1 if first else 0}"] + inputs + weights + biases + outputs + scaled
    rows.append(" ".join(columns))


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps over the other input vector run before the recorded
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
    return arms[0].layer.hw.pp_delay


def run_saturation(rows):
    """Append the positive-clamp block, which is what makes WIDTH observable.

    Weight and bias sit at the polarity's high rail, so the lane popcount is
    ENTRY over the high input block and the bias alone over the low one. With
    SCALE_S below ENTRY the high block charges both scaled arms' accumulators
    onto their positive clamp at 2**(WIDTH-1) - 1, and the low block drains
    them: the stored charge is how many further cycles an arm keeps emitting, so
    a narrower WIDTH stops emitting sooner and its output column differs.
    """
    arms = [arm('unipolar', rail='pos'), arm('bipolar', rail='pos')]
    charge = [1] * (2 * TIMESTEPS) + [0] * TIMESTEPS
    peak = [0, 0]
    for step, index in enumerate(charge):
        emit_row(rows, arms, index, step == 0)
        for slot, one in enumerate(arms):
            peak[slot] = max(peak[slot], int(one.layer_s.acc.accumulator.reshape(-1)[0]))
    # One carry is subtracted after the clamp, so a clamped accumulator is
    # retained at acc_max - SCALE_S.
    for slot, one in enumerate(arms):
        acc_max = one.layer_s.acc.acc_max
        assert peak[slot] == acc_max - SCALE_S, \
            f'the {one.polarity} scaled accumulator peaked at {peak[slot]}, short of {acc_max}'


def run_negative(rows):
    """Append the negative-clamp block, which is what makes ACC_LO observable.

    The positive block cannot reach the negative clamp: its high rail bias keeps
    an addend on every cycle, and only a bipolar accumulator moves down at all,
    by the offset (ENTRY - SCALE) / 2 that a scale below the fan-in creates. So
    this block rails the bias low as well, which takes the popcount to zero over
    the low input block, and holds it there long enough for the bipolar scaled
    arm to fall onto -2**(WIDTH-1). The high input block then recharges it: how
    long the arm stays silent on the way back up is what the clamp sets, so a
    shallower ACC_LO starts emitting sooner and its column differs.

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
            bottom[slot] = min(bottom[slot], int(one.layer_s.acc.accumulator.reshape(-1)[0]))
    acc_min = arms[1].layer_s.acc.acc_min
    assert bottom[1] == acc_min, \
        f'the bipolar scaled accumulator bottomed at {bottom[1]}, short of {acc_min}'
    assert bottom[0] == 0, \
        f'the unipolar scaled accumulator went negative, to {bottom[0]}'


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["rst in_u in_b w_u w_b b_u b_b out_u out_u_nb out_b out_b_nb out_u_s out_b_s"]
    pp_delay = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=TIMESTEPS // 2)
    saturation_start = len(rows)
    run_saturation(rows)
    negative_start = len(rows)
    run_negative(rows)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:1 + 3 * TIMESTEPS], \
        'reset() did not restore the opening accumulator state'
    # A saturation block only makes its clamp observable if the clamped column
    # moves, which is what the recovery window on either side of the clamp does.
    for start, stop, label in [(saturation_start, negative_start, 'positive'),
                               (negative_start, len(rows), 'negative')]:
        for column, polarity in [(-2, 'unipolar'), (-1, 'bipolar')]:
            block = [row.split()[column] for row in rows[start:stop]]
            if label == 'negative' and polarity == 'unipolar':
                continue
            assert len(set(block)) > 1, \
                f'the {label} block never moved the {polarity} scaled column'
    # Bipolar p = (x + 1) / 2 over the unipolar grid would reproduce the unipolar
    # columns exactly, leaving the bipolar arm no distinct stimulus.
    columns = [row.split() for row in rows[1:]]
    assert [row[1] for row in columns] != [row[2] for row in columns], \
        'bipolar input stimulus is identical to unipolar'
    assert [row[3] for row in columns] != [row[4] for row in columns], \
        'bipolar weight stimulus is identical to unipolar'

    defines = [("IN_FEATURES", IN_FEATURES), ("LANES", LANES), ("WIDTH", ACC_WIDTH),
               ("SCALE", IN_FEATURES + 1), ("SCALE_NB", IN_FEATURES), ("SCALE_S", SCALE_S),
               ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    PARAMS.write_text("".join(f"`define GEN_{name} {value}\n" for name, value in defines))
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} ("
        + ", ".join(f"GEN_{name}={value}" for name, value in defines)
        + ")"
    )


if __name__ == "__main__":
    main()
