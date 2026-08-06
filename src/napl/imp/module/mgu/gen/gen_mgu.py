"""
Generate golden test vectors for the mgu module RTL straight from napl's
functional Python model (napl.sim.module.mgu) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

mgu is the first recurrent cell in the module tree. Reading forward() term by
term gives the decomposition the RTL instantiates, all of it existing circuits:

    fg_in = fg_ug_tanh(cat(hx_spike, input_spike))  linear_bipolar,   scale 1
    fg    = fg_sigmoid(fg_in)                       sigmoid_hard
    fg_hx = fg_hx_mul(fg, hx_value)                 mul_ugemm_bipolar
    ng    = ng_ug_tanh(cat(fg_hx, input_spike))     linear_bipolar,   scale 1
    fg_ng = fg_ng_mul(fg, ng)                       mul_ugemm_sr_bipolar
    out   = hy_add(ng + (1 - fg_ng) + fg_hx, 3)     add_any_bipolar,  scale 1

The output adder is a three-addend add_any over ng, 1 - fg_ng, and fg_hx, not a
two-operand reduction, and both gate layers take the SAME cat() width
LANES + IN_SIZE: the fg gate over cat(hx_spike, input_spike) and the n gate over
cat(fg_hx, input_spike). Every lane's gate row reads every lane's hidden
feature, so each gate is one linear_bipolar over all lanes rather than a
per-lane replication.

Held versus streamed operands: the gate weights and biases are re-encoded from
external tensors every timestep, so they are captured from the model's own
encoders with forward hooks and driven on ports. hx_value is a registered
buffer that mul_ugemm reads as a number, reachable by no hook, so it is read
straight from the model's state (cell.hx_value) and emitted as the fixed-point
code the RTL comparator takes. The codes are drawn on the 1/LEN grid so the
model's float compare and the RTL's integer compare agree exactly.

Two DUT configurations share each row, one column of expected output each:
  out     -- both gate biases present
  out_nb  -- neither gate bias present

Both `mul_ugemm_bipolar` and `mul_ugemm_sr_bipolar` $readmemb their sequence ROM
from a path relative to the simulation cwd, which is module/mgu/, so this file
writes vec/mul_ugemm_rom.hex and vec/mul_ugemm_sr_rom.hex from the model's own
tables.

Three sequences follow each other: one over the first input pair, one over the
second, then the first replayed after the second has dirtied the state and
reset() has cleared it. The replay is the state check that matters here, since
the cell is stateful and mul_ugemm_sr's shift register is its stateful heart:
DIRTY_STEPS is odd and coprime with the 2**SR_WIDTH register depth, so the
register head lands on an odd offset and the stored contents cannot alias back
to the alternating reset pattern by accident. A reset that restored only the
accumulators and sequence indices, leaving the register rotated, would produce a
different first sequence and fail the equality assertion below.

Output: ../vec/mgu.vec, one line per timestep:

    <rst> <in> <hx> <wf> <bf> <wn> <bn> <hxv> <out> <out_nb>

`in` is IN_SIZE binary digits, `hx`, `bf`, `bn`, `out`, and `out_nb` are LANES
digits, `wf` and `wn` are LANES*(LANES+IN_SIZE) digits with lane l, feature f at
[l*(LANES+IN_SIZE) + f], and `hxv` is LANES*(SEQ_WIDTH+1) digits with lane l at
[l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]. All are highest bit index first.

Sizing values come from this file only and are emitted into ../vec/mgu_params.vh,
so the testbench elaborates the RTL at the model's configuration and the two
cannot drift. The header also records pp_delay.

check_mapping() translates this module's own mapping.yaml entry and requires the
resolved parameters to equal the ones used here, for both elaborated
configurations, so the co-simulation gates the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_mgu.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import mgu
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "mgu.vec"
PARAMS = VEC_DIR / "mgu_params.vh"

# Shapes and codec mirror tests/module/test_mgu.py, at batch 1: LANES is one
# lane per hidden unit and the RTL ports carry no batch dimension.
IN_SIZE = 4
LANES = 3
TIMESTEP = 256
TIMESTEPS = TIMESTEP
ACC_WIDTH = 10
SR_WIDTH = 6
IN_FEATURES = LANES + IN_SIZE
SEQ_WIDTH = 8
LEN = 2 ** SEQ_WIDTH
# Odd, and coprime with the 2**SR_WIDTH shift-register depth, so the dirty run
# leaves the register head off its reset position and its contents off the
# alternating reset pattern.
DIRTY_STEPS = 77
# Negative-rail block: the forget gate falls at the offset 3.5 per silent cycle
# with both gate biases railed low, so 147 cycles reach -2**(WIDTH-1) = -512 and
# the no-bias cell, falling at 3, needs 171. The recharge runs at 3.5 per cycle,
# so a lane starting from -512 stays silent for 147 cycles and one held at a
# halved clamp starts emitting halfway through that.
NEG_DRAIN = 180
NEG_CHARGE = 160
# Positive-rail block: with both gate biases railed high the forget gate's
# popcount is ENTRY = IN_FEATURES + 1 on a railed-high input, so its accumulator
# climbs at ENTRY - offset - 1 = 3.5 per cycle and reaches 2**(WIDTH-1) - 1 = 511
# in 146 cycles. The railed-low block then drains it at 2.5 per cycle, the bias
# addend being all that is left.
SAT_CHARGE = 146
SAT_DRAIN = 105

POLARITY = 'bipolar'


def grid(count, shape, offset):
    """A deterministic bipolar operand tensor spanning most of [-1, 1]."""
    steps = torch.linspace(-0.75, 0.75, count, dtype=global_config.ntype)
    return steps.roll(offset).reshape(shape)


def hx_codes():
    """Fixed-point codes in [0, LEN] for the held hidden value, one per lane."""
    return [(index * 53 + 29) % (LEN + 1) for index in range(LANES)]


def hx_value():
    """The held hidden value on the 1/LEN grid, so both compares agree exactly.

    mul_ugemm compares (hx_value + 1) / 2 against a float number-sequence entry
    while the RTL compares the fixed-point code against the same entry scaled by
    LEN. code / 128 - 1 is exact in float32 for a power-of-two LEN, so the two
    compares take the same branch on every index.
    """
    values = [code / (LEN // 2) - 1.0 for code in hx_codes()]
    return torch.tensor(values, dtype=global_config.ntype).reshape(1, LANES)


def as_binary(bits):
    """Render a bit list as a Verilog %b string, highest bit index first."""
    return "".join(str(int(bit)) for bit in reversed(bits))


def bits_of(spike):
    """Flatten a spike tensor to a python bit list in lane order."""
    return spike.reshape(-1).to(torch.int64).tolist()


def hx_value_bits(cell):
    """The held operand column, read from the model's own hx_value buffer.

    No forward hook reaches it: mul_ugemm consumes hx_value as a call argument
    rather than as an encoder output, so the model's state is the only source.
    """
    values = cell.hx_value.detach().reshape(-1)
    assert values.numel() == LANES, f'hx_value holds {values.numel()} lanes, expected {LANES}'
    bits = []
    for lane in range(LANES):
        scaled = ((values[lane].item() + 1.0) / 2.0) * LEN
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f'hx_value[{lane}] is off the 1/{LEN} grid'
        assert 0 <= code <= LEN, f'hx_value[{lane}] code {code} outside [0, {LEN}]'
        bits.extend((code >> bit) & 1 for bit in range(SEQ_WIDTH + 1))
    return bits


class capture:
    """Forward hook that keeps the most recent output of an encoder."""


    def __init__(self, module):
        self.spike = None
        module.register_forward_hook(self)


    def __call__(self, module, args, output):
        self.spike = output


class arm:
    """The input encoders plus the with-bias and no-bias cell models."""


    def __init__(self, rail=None):
        codec_in = {'polarity': POLARITY, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 1}
        codec_hx = dict(codec_in, dim=2)
        self.config = {'polarity': POLARITY, 'timestep': TIMESTEP, 'generator': 'sobol',
                       'width': ACC_WIDTH, 'depth_ismul': SR_WIDTH}
        require_seeded_sys(codec_in, codec_hx, self.config)
        if rail is not None:
            # Weights at +1 give the forget gate its widest per-timestep swing:
            # every product spike follows the input spike, so the gate popcount
            # is IN_FEATURES on a railed-high input and the bias alone on a
            # railed-low one. rail='pos' rails the biases high too, which keeps
            # one addend on every cycle; rail='neg' rails them low, which takes
            # the popcount to zero and lets the offset drive the accumulator onto
            # its negative clamp.
            level = 1.0 if rail == 'pos' else -1.0
            self.weight_f = torch.ones(LANES, IN_FEATURES, dtype=global_config.ntype)
            self.weight_n = torch.ones(LANES, IN_FEATURES, dtype=global_config.ntype)
            self.bias_f = torch.full((LANES,), level, dtype=global_config.ntype)
            self.bias_n = torch.full((LANES,), level, dtype=global_config.ntype)
            self.values = [(torch.full((1, IN_SIZE), value, dtype=global_config.ntype),
                            torch.full((1, LANES), value, dtype=global_config.ntype))
                           for value in (-1.0, 1.0)]
        else:
            self.weight_f = grid(LANES * IN_FEATURES, (LANES, IN_FEATURES), 3)
            self.weight_n = grid(LANES * IN_FEATURES, (LANES, IN_FEATURES), 11)
            self.bias_f = grid(LANES, (LANES,), 1)
            self.bias_n = grid(LANES, (LANES,), 2)
            self.values = [(grid(IN_SIZE, (1, IN_SIZE), offset),
                            grid(LANES, (1, LANES), offset + 1))
                           for offset in (0, 5)]
        self.hx_value = hx_value()
        self.enc_in = encode(codec_in)
        self.enc_hx = encode(codec_hx)
        self.cell = mgu(self.weight_f, self.bias_f, self.weight_n, self.bias_n,
                        self.hx_value, self.config)
        self.cell_nb = mgu(self.weight_f, None, self.weight_n, None,
                           self.hx_value, self.config)
        self.weight_f_spike = capture(self.cell.fg_ug_tanh.w_encoder)
        self.weight_n_spike = capture(self.cell.ng_ug_tanh.w_encoder)
        self.bias_f_spike = capture(self.cell.fg_ug_tanh.b_encoder)
        self.bias_n_spike = capture(self.cell.ng_ug_tanh.b_encoder)
        self.weight_f_spike_nb = capture(self.cell_nb.fg_ug_tanh.w_encoder)
        self.weight_n_spike_nb = capture(self.cell_nb.ng_ug_tanh.w_encoder)
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this run's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Both elaborated configurations are translated, since dropping
        the biases changes HAS_BIAS_F and HAS_BIAS_N.
        """
        node = {"class": "mgu",
                "config": {"weight_f": self.weight_f, "bias_f": self.bias_f,
                           "weight_n": self.weight_n, "bias_n": self.bias_n,
                           "hx_value": self.hx_value,
                           "config": self.config, "lanes": LANES}}
        expected = {"LANES": LANES, "IN_SIZE": IN_SIZE, "WIDTH": ACC_WIDTH,
                    "SEQ_WIDTH": SEQ_WIDTH, "SR_WIDTH": SR_WIDTH,
                    "HAS_BIAS_F": 1, "HAS_BIAS_N": 1}
        binding = translate_node(node)
        assert binding.rtl_module == "mgu_bipolar", binding.rtl_module
        assert binding.parameters == expected, \
            f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"

        node_nb = dict(node, config=dict(node["config"], bias_f=None, bias_n=None))
        expected_nb = dict(expected, HAS_BIAS_F=0, HAS_BIAS_N=0)
        binding_nb = translate_node(node_nb)
        assert binding_nb.parameters == expected_nb, \
            f"mapping.yaml {binding_nb.rtl_module} resolves {binding_nb.parameters}"


    def reset(self):
        """Restart both input encoders and both cells."""
        self.enc_in.reset()
        self.enc_hx.reset()
        self.cell.reset()
        self.cell_nb.reset()


    def step(self, index):
        """Advance one timestep over input pair `index`; return the row columns."""
        input_value, hx_spike_value = self.values[index]
        in_spike = self.enc_in(input_value)
        hx_spike = self.enc_hx(hx_spike_value)
        out = self.cell(in_spike, hx_spike)
        out_nb = self.cell_nb(in_spike, hx_spike)
        weight_f_bits = bits_of(self.weight_f_spike.spike)
        weight_n_bits = bits_of(self.weight_n_spike.spike)
        assert weight_f_bits == bits_of(self.weight_f_spike_nb.spike), \
            'the two cells encoded different forget-gate weight spikes'
        assert weight_n_bits == bits_of(self.weight_n_spike_nb.spike), \
            'the two cells encoded different new-gate weight spikes'
        return (bits_of(in_spike), bits_of(hx_spike),
                weight_f_bits, bits_of(self.bias_f_spike.spike),
                weight_n_bits, bits_of(self.bias_n_spike.spike),
                hx_value_bits(self.cell), bits_of(out), bits_of(out_nb))


def write_rom(cell):
    """Emit both sequence ROMs the composed multipliers read at simulation time.

    mul_ugemm_bipolar reads vec/mul_ugemm_rom.hex and mul_ugemm_sr_bipolar reads
    vec/mul_ugemm_sr_rom.hex, both relative to the simulation cwd. The tables are
    the model's own, so a ROM that drifted from the model would change the
    compared outputs.
    """
    num_seq = cell.fg_hx_mul.num_seq.detach().float().reshape(-1)
    assert num_seq.numel() == LEN, f'num_seq length {num_seq.numel()} != LEN {LEN}'
    lines = []
    for index in range(LEN):
        scaled = num_seq[index].item() * LEN
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f'num_seq[{index}] off the 1/{LEN} grid'
        lines.append(f"{code:0{SEQ_WIDTH}b}")
    (VEC_DIR / "mul_ugemm_rom.hex").write_text("\n".join(lines) + "\n")

    rng_seq = cell.fg_ng_mul.rng_seq.detach().reshape(-1)
    depth = 2 ** SR_WIDTH
    assert rng_seq.numel() == depth, f'rng_seq length {rng_seq.numel()} != depth {depth}'
    (VEC_DIR / "mul_ugemm_sr_rom.hex").write_text(
        "".join(f"{int(value.item()):0{SR_WIDTH}b}\n" for value in rng_seq)
    )


def run_sequence(rows, index, dirty=0):
    """Append one reset-to-reset sequence of vectors.

    `dirty` timesteps over the other input pair run before the recorded sequence
    and are followed by reset(), so the recorded block starts from a cell whose
    accumulators, sequence indices, and shift register all held state. Its
    outputs must equal the same block recorded from a fresh cell, which the
    testbench checks by pulsing i_rst_n on the opening rst=1 row.
    """
    one = arm()
    one.check_mapping()
    if dirty:
        for _ in range(dirty):
            one.step(1 - index)
        one.reset()

    for timestep in range(TIMESTEPS):
        columns = [f"{1 if timestep == 0 else 0}"]
        columns += [as_binary(bits) for bits in one.step(index)]
        rows.append(" ".join(columns))
    return one.cell


def run_saturation(rows):
    """Append the accumulator-saturation block, which is what makes WIDTH observable.

    The gate popcount swings between IN_FEATURES and 0, so the forget gate's
    accumulator charges over the railed-high block and drains over the railed-low
    one. How long a lane keeps emitting on the way down is the charge it stored,
    and a narrower WIDTH clamps that charge, so its output column differs.

    mul_ugemm's sequence index has no modulo and is read every timestep, so one
    run may not exceed 2**SEQ_WIDTH timesteps on either index. Within that budget
    the charge reaches the elaborated clamp at 2**(WIDTH-1) - 1 itself, so a
    clamped accumulator is retained at that value less the gate scale of 1.
    The no-bias cell's gate carries one addend fewer, so it climbs at 3 rather
    than 3.5 and stops short of the clamp; the with-bias cell is what this block
    pins WIDTH with.
    """
    one = arm(rail='pos')
    charge = [1] * SAT_CHARGE + [0] * SAT_DRAIN
    peak = 0
    peak_nb = 0
    for step, index in enumerate(charge):
        columns = [f"{1 if step == 0 else 0}"]
        columns += [as_binary(bits) for bits in one.step(index)]
        peak = max(peak, int(one.cell.fg_ug_tanh.acc.accumulator.reshape(-1).max()))
        peak_nb = max(peak_nb, int(one.cell_nb.fg_ug_tanh.acc.accumulator.reshape(-1).max()))
        rows.append(" ".join(columns))
    acc_max = one.cell.fg_ug_tanh.acc.acc_max
    narrower = 2 ** (ACC_WIDTH - 2) - 1
    assert peak == acc_max - 1, \
        f'the forget-gate accumulator peaked at {peak}, short of the clamp at {acc_max}'
    assert peak_nb > narrower, \
        f'the no-bias forget-gate accumulator peaked at {peak_nb}, inside a {ACC_WIDTH - 1}-bit clamp'
    return peak


def run_negative(rows):
    """Append the negative-rail block, which is what makes ACC_LO observable.

    The positive block cannot reach the negative clamp: its railed-high bias
    keeps one addend on every cycle. This block rails both gate biases low
    instead, so a railed-low input takes the forget gate's popcount to zero and
    its accumulator falls at the offset (ENTRY - SCALE) / 2 = 3.5 per cycle,
    which reaches -2**(WIDTH-1) = -512 in 147 cycles. The railed-high block then
    recharges it at IN_FEATURES - offset = 3.5 per cycle: how long a lane stays
    silent on the way back up is what the clamp sets, so a shallower ACC_LO
    starts emitting sooner and the compared column differs.

    The gate scale is 1 and the gate ENTRY is IN_FEATURES + 1 with the bias and
    IN_FEATURES without, so the no-bias cell falls at 3 per cycle and takes 171
    cycles; the drain covers both. The output adder is the other accumulator in
    the cell, at ENTRY 3 and scale 1: it drifts by at most 1 per cycle, so no run
    inside the 2**SEQ_WIDTH sequence-index budget can move it past +-256, and
    neither of its clamps is reachable at this WIDTH. The assertions below record
    both reaches rather than assume them.
    """
    one = arm(rail='neg')
    drain = [0] * NEG_DRAIN + [1] * NEG_CHARGE
    bottom = [0, 0]
    gate_n = 0
    output_reach = 0
    for step, index in enumerate(drain):
        columns = [f"{1 if step == 0 else 0}"]
        columns += [as_binary(bits) for bits in one.step(index)]
        for slot, cell in enumerate([one.cell, one.cell_nb]):
            bottom[slot] = min(bottom[slot], float(cell.fg_ug_tanh.acc.accumulator.reshape(-1).min()))
            gate_n = min(gate_n, float(cell.ng_ug_tanh.acc.accumulator.reshape(-1).min()))
            output_reach = max(output_reach, float(cell.hy_add.accumulator.reshape(-1).abs().max()))
        rows.append(" ".join(columns))
    acc_min = one.cell.fg_ug_tanh.acc.acc_min
    for slot, label in enumerate(['bias', 'nobias']):
        assert bottom[slot] == acc_min, \
            f'the {label} forget-gate accumulator bottomed at {bottom[slot]}, short of {acc_min}'
    # The n gate is driven by fg_hx rather than by the input directly, so its own
    # popcount is not railed and it stops short of the clamp; it does pass the
    # clamp of the next narrower width, which is what its column pins.
    assert gate_n < -2 ** (ACC_WIDTH - 2), \
        f'the new-gate accumulator bottomed at {gate_n}, inside a {ACC_WIDTH - 1}-bit clamp'
    assert output_reach < 2 ** (ACC_WIDTH - 2), \
        f'the output adder reached {output_reach}, past the clamp of a narrower width'
    return bottom[0]


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["rst in hx wf bf wn bn hxv out out_nb"]
    cell = run_sequence(rows, 0)
    run_sequence(rows, 1)
    run_sequence(rows, 0, dirty=DIRTY_STEPS)
    peak = run_saturation(rows)
    depth = run_negative(rows)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:1 + 3 * TIMESTEPS], \
        'reset() did not restore the opening cell state'
    # The two input pairs must be distinct stimulus, or the replay above would
    # compare a sequence against a copy of itself.
    assert rows[1:1 + TIMESTEPS] != rows[1 + TIMESTEPS:1 + 2 * TIMESTEPS], \
        'the two input pairs produce identical vectors'
    write_rom(cell)

    pp_delay = cell.hw.pp_delay
    defines = [("LANES", LANES), ("IN_SIZE", IN_SIZE), ("WIDTH", ACC_WIDTH),
               ("SEQ_WIDTH", SEQ_WIDTH), ("SR_WIDTH", SR_WIDTH),
               ("PP_DELAY", pp_delay), ("VECTORS", len(rows) - 1)]
    PARAMS.write_text("".join(f"`define GEN_{name} {value}\n" for name, value in defines))
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} ("
        + ", ".join(f"GEN_{name}={value}" for name, value in defines)
        + f", excursion {peak} / {depth})"
    )


if __name__ == "__main__":
    main()
