"""
Generate golden test vectors for the linear_pc module RTL straight from napl's
functional Python model (napl.sim.module.linear_pc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

linear_pc reduces one lane per output feature and publishes the lane's parallel
count instead of a spike. The count is the same popcount linear builds and hands
to add_any:

    unipolar   count = sum_f x_f * w_f                      = popcount(AND(x, w))
    bipolar    count = sum_f x_f*w_f + sum_f (1-x_f)(1-w_f)
                     = sum_f (1 - x_f - w_f + 2*x_f*w_f)    = popcount(XNOR(x, w))

with the bias spike added as one more addend. So linear_pc is linear minus the
add_any stage, and the RTL lane is IN_FEATURES mul_gaines_<polarity> cells plus
the optional bias bit feeding a popcount:

    mul_gaines_<polarity>  -- one per (output feature, input feature)
    popcount               -- one per output feature, ENTRY addends in, COUNT_W bits out

The popcount is written in the module because the operation layer holds no
standalone popcount circuit: add_any bundles it with the scaled accumulator
linear_pc does not have.

The model re-encodes the weight and the bias from externally held tensors every
timestep, so both arrive on RTL ports rather than being rebuilt in hardware. The
spikes are captured from the model's own encoders with forward hooks, so the
columns are the exact bits the model reduced.

Four DUT configurations share each row, one column of expected output each:
  out_u     -- unipolar, with bias   (entry = in_features + 1)
  out_u_nb  -- unipolar, no bias     (entry = in_features)
  out_b     -- bipolar,  with bias
  out_b_nb  -- bipolar,  no bias

The class takes no scale or accumulator width, so there is no scaled variant to
elaborate: the counter has no accumulator to scale.

The unipolar and the bipolar value grids span different ranges, so the encoded
bipolar rate p = (x + 1) / 2 is a different stimulus from the unipolar rate.

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded streams. Three sequences follow
each other: one over the first input vector, one over the second, then the first
replayed after the second has advanced the encoders and reset() has restarted
them, which must reproduce the first sequence bit for bit. linear_pc holds no
accumulator, so the RTL is stateless and carries no reset port: that replay checks
the Python encoders the vectors are drawn from, not RTL reset behavior.

Output: ../vec/linear_pc.vec, one line per timestep:

    <in_u> <in_b> <w_u> <w_b> <b_u> <b_b> <out_u> <out_u_nb> <out_b> <out_b_nb>

`in_*` is IN_FEATURES binary digits, MSB first, so input feature f occupies
i_input_spike[f]. `w_*` is LANES*IN_FEATURES digits with lane l, feature f at
i_weight[l*IN_FEATURES + f]. `b_*` is LANES digits, lane l the output feature l.

Each `out_*` column is a *count bus*, not one bit per lane: lane l holds a
COUNT_W-bit unsigned count at o_out[l*COUNT_W +: COUNT_W], so the column is
LANES*COUNT_W digits, highest bit index first like every other column. The testbench checks that character count against LANES*COUNT_W. That check is an
echo of COUNT_W rather than a check on it, since count_width() below sizes both
the column and the elaborated parameter; COUNT_W is pinned by the RTL's
elaboration guard against clog2(ENTRY + 1) and by check_mapping() against the
mapping.yaml expression.

Sizing values come from this file only and are emitted into
../vec/linear_pc_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay and the
row count, which the testbench compares against the rows it consumed.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_linear_pc.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import linear_pc
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "linear_pc.vec"
PARAMS = VEC_DIR / "linear_pc_params.vh"

# Shapes and codec mirror tests/module/test_linear_pc.py.
IN_FEATURES = 16
LANES = 8
TIMESTEP = 256
TIMESTEPS = TIMESTEP

# Bipolar values run over a narrower range than unipolar ones, so the encoded
# bipolar rate (x + 1) / 2 differs from the unipolar rate x.
RANGE = {'unipolar': (0.0, 1.0), 'bipolar': (-1.0, 0.75)}


def clog2(value):
    """Bits needed to hold 0..value-1, matching the RTL clog2 function."""
    width = 0
    remaining = value - 1
    while remaining > 0:
        width += 1
        remaining >>= 1
    return width


def count_width(has_bias):
    """Bits per lane count: clog2(entry + 1) for entry = in_features + has_bias."""
    return clog2(IN_FEATURES + int(has_bias) + 1)


# The no-bias arm's entry is IN_FEATURES, a power of two, so clog2(entry) is one
# bit narrower than clog2(entry + 1): that arm is what pins the + 1, and the
# with-bias arm's entry alone would not tell the two apart.
assert IN_FEATURES & (IN_FEATURES - 1) == 0, 'IN_FEATURES must be a power of two'
assert count_width(False) == clog2(IN_FEATURES) + 1, 'the count width lost its + 1'



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
    """One polarity's input encoder plus its with-bias and no-bias counter models."""


    def __init__(self, polarity):
        self.polarity = polarity
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 2}
        require_seeded_sys(codec, config)
        self.config = config
        self.weight = grid(LANES * IN_FEATURES, (LANES, IN_FEATURES), polarity, 3)
        self.bias = grid(LANES, (LANES,), polarity, 1)
        self.values = [grid(IN_FEATURES, (IN_FEATURES,), polarity, offset)
                       for offset in (0, 5)]
        self.enc = encode(codec)
        self.layer = linear_pc(self.weight, self.bias, config)
        self.layer_nb = linear_pc(self.weight, None, config)
        self.weight_spike = capture(self.layer.w_encoder)
        self.weight_spike_nb = capture(self.layer_nb.w_encoder)
        self.bias_spike = capture(self.layer.b_encoder)
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Both elaborated configurations are translated: with bias and
        without, which change ENTRY and so the count width.
        """
        node = {"class": "linear_pc",
                "config": {"weight": self.weight, "bias": self.bias,
                           "config": self.config, "lanes": LANES}}
        expected = {"IN_FEATURES": IN_FEATURES, "LANES": LANES, "HAS_BIAS": 1,
                    "COUNT_W": count_width(True)}
        binding = translate_node(node)
        assert binding.rtl_module == f"linear_pc_{self.polarity}", binding.rtl_module
        assert binding.parameters == expected, \
            f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"

        node_nb = dict(node, config=dict(node["config"], bias=None))
        expected_nb = dict(expected, HAS_BIAS=0, COUNT_W=count_width(False))
        binding_nb = translate_node(node_nb)
        assert binding_nb.parameters == expected_nb, \
            f"mapping.yaml {binding_nb.rtl_module} resolves {binding_nb.parameters}"


    def reset(self):
        """Restart the input encoder and every layer."""
        self.enc.reset()
        self.layer.reset()
        self.layer_nb.reset()


    def step(self, index):
        """Advance one timestep over input vector `index`; return the row columns."""
        spike = self.enc(self.values[index])
        out = self.layer(spike)
        out_nb = self.layer_nb(spike)
        weight_bits = bits_of(self.weight_spike.spike)
        assert weight_bits == bits_of(self.weight_spike_nb.spike), \
            'the two layers encoded different weight spikes'
        return (bits_of(spike), weight_bits, bits_of(self.bias_spike.spike),
                count_bits(out, count_width(True), IN_FEATURES + 1),
                count_bits(out_nb, count_width(False), IN_FEATURES))


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors for both polarities.

    `dirty` timesteps over the other input vector run before the recorded
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
        outputs = []
        for one in arms:
            in_bits, weight_bits, bias_bits, out_bits, out_nb_bits = one.step(index)
            inputs.append(as_binary(in_bits))
            weights.append(as_binary(weight_bits))
            biases.append(as_binary(bias_bits))
            outputs.append(as_binary(out_bits))
            outputs.append(as_binary(out_nb_bits))
        rows.append(" ".join(inputs + weights + biases + outputs))
    return arms[0].layer.hw.pp_delay


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["in_u in_b w_u w_b b_u b_b out_u out_u_nb out_b out_b_nb"]
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

    defines = [("IN_FEATURES", IN_FEATURES), ("LANES", LANES),
               ("COUNT_W", count_width(True)), ("COUNT_W_NB", count_width(False)),
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
