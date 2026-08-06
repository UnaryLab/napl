"""
Generate golden test vectors for the linear_ugemm module RTL straight from napl's
functional Python model (napl.sim.module.linear_ugemm) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

linear_ugemm is stateful: every (output, input) weight bit comes from a
conditional spike generator whose sequence index advances only on an input spike,
and every output feature owns a scaled unary accumulator. The RTL composes the
operation-layer circuits that implement exactly those pieces:

    mul_ugemm_<polarity>  -- one per (output feature, input feature) weight bit
    encode                -- one per output feature, the free-running bias bit
    add_any_<polarity>    -- one per output feature, the scaled accumulator

Both `mul_ugemm_*` and `encode` $readmemb their number-sequence ROM from a path
relative to the simulation cwd (module/linear_ugemm/), so this file writes
vec/mul_ugemm_rom.hex and vec/encode_rom.hex with the sequence of the built model.
Both tables are the same sequence, which is why the bias comparator can be the
encode circuit: linear_ugemm indexes self.mul.num_seq with the free-running
timestep for the bias bit.

Sequence-index bound: mul_ugemm advances seq_idx without a modulo, so a run stays
valid for at most 2**ceil(log2(timestep)) timesteps. TIMESTEPS is pinned to that
bound (LEN), and every driven block is TIMESTEPS long or shorter.

Four DUT configurations share each row, one column of expected output each:
  out_u     -- unipolar, with bias   (entry = scale = in_features + 1)
  out_u_nb  -- unipolar, no bias     (entry = scale = in_features)
  out_b     -- bipolar,  with bias
  out_b_nb  -- bipolar,  no bias

Polarity selects the circuit here (separate `_unipolar` / `_bipolar` modules), so
the two polarities run as parallel columns rather than sequential blocks: each row
drives both polarity DUTs with their own encoded input stream. Two sequences
follow each other: a fresh stream, then the same stream replayed after half a
stream has dirtied the sequence indices and the accumulators and reset() has
cleared them, which must reproduce the first sequence bit for bit.

Output: ../vec/linear_ugemm.vec, one line per timestep:

    <rst> <in_u_bits> <in_b_bits> <out_u> <out_u_nb> <out_b> <out_b_nb>

`in_*_bits` is IN_FEATURES binary digits, MSB first, so input feature f occupies
i_input_spike[f]. `out_*` is LANES binary digits, MSB first, with lane l the
output feature l.

Weight and bias operands are held fixed-point codes, so they go to
../vec/linear_ugemm_operand_u.hex and _b.hex (LANES*(IN_FEATURES+1) lines each:
IN_FEATURES weight codes then the bias code, per lane) instead of a vector column.

Sizing values come from this file only and are emitted into
../vec/linear_ugemm_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay.

arm.check_mapping() translates this module's own mapping.yaml entry and requires
the resolved parameters to equal the ones used here, so the co-simulation gates
the mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_linear_ugemm.py
"""
import math
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import linear_ugemm
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC_DIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VEC_DIR / "linear_ugemm.vec"
PARAMS = VEC_DIR / "linear_ugemm_params.vh"

# Shapes and codec mirror tests/module/test_linear_ugemm.py.
IN_FEATURES = 16
LANES = 8
TIMESTEP = 256
ACC_WIDTH = 12

SEQ_WIDTH = math.ceil(math.log2(TIMESTEP))
LEN = 2 ** SEQ_WIDTH
# mul_ugemm's sequence index has no modulo, so a run may not exceed LEN steps.
TIMESTEPS = min(TIMESTEP, LEN)


def operand_codes(count, stride, offset):
    """Return `count` fixed-point codes in [0, LEN] on the 1/LEN operand grid."""
    return [(i * stride + offset) % (LEN + 1) for i in range(count)]


WEIGHT_CODES = operand_codes(LANES * IN_FEATURES, 29, 5)
BIAS_CODES = operand_codes(LANES, 53, 7)
INPUT_CODES = operand_codes(IN_FEATURES, 71, 13)


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
        codec = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol', 'dim': 2}
        config = {'polarity': polarity, 'timestep': TIMESTEP, 'generator': 'sobol',
                  'dim': 1, 'scale': None, 'width': ACC_WIDTH}
        require_seeded_sys(codec, config)
        weight = tensor(WEIGHT_CODES, (LANES, IN_FEATURES), polarity)
        bias = tensor(BIAS_CODES, (LANES,), polarity)
        self.value = tensor(INPUT_CODES, (IN_FEATURES,), polarity)
        self.enc = encode(codec)
        self.layer = linear_ugemm(weight, bias, config)
        self.layer_nb = linear_ugemm(weight, None, config)
        self.config = config
        self.weight = weight
        self.bias = bias
        self.reset()


    def check_mapping(self):
        """Check mapping.yaml resolves this arm's own sizing parameters.

        The vectors below are built straight from the Python model, so without
        this the mapping entry could drift from the hardware the co-simulation
        verifies. Both elaborated configurations are translated: with bias and
        without, which change ENTRY and so the default scale.
        """
        node = {"class": "linear_ugemm",
                "config": {"weight": self.weight, "bias": self.bias,
                           "config": self.config, "lanes": LANES}}
        expected = {"IN_FEATURES": IN_FEATURES, "LANES": LANES, "SEQ_WIDTH": SEQ_WIDTH,
                    "WIDTH": ACC_WIDTH, "HAS_BIAS": 1, "SCALE": IN_FEATURES + 1}
        binding = translate_node(node)
        assert binding.rtl_module == f"linear_ugemm_{self.polarity}", binding.rtl_module
        assert binding.parameters == expected, \
            f"mapping.yaml {binding.rtl_module} resolves {binding.parameters}"

        node_nb = dict(node, config=dict(node["config"], bias=None))
        expected_nb = dict(expected, HAS_BIAS=0, SCALE=IN_FEATURES)
        binding_nb = translate_node(node_nb)
        assert binding_nb.parameters == expected_nb, \
            f"mapping.yaml {binding_nb.rtl_module} resolves {binding_nb.parameters}"


    def reset(self):
        """Restart the encoder and both layers."""
        self.enc.reset()
        self.layer.reset()
        self.layer_nb.reset()


    def step(self):
        """Advance one timestep; return (input bits, out bits, no-bias out bits)."""
        spike = self.enc(self.value)
        out = self.layer(spike)
        out_nb = self.layer_nb(spike)
        return bits_of(spike), bits_of(out), bits_of(out_nb)


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
        columns = [f"{1 if timestep == 0 else 0}"]
        outputs = []
        for one in arms:
            in_bits, out_bits, out_nb_bits = one.step()
            columns.append(as_binary(in_bits))
            outputs.append(as_binary(out_bits))
            outputs.append(as_binary(out_nb_bits))
        rows.append(" ".join(columns + outputs))
    return arms[0].layer


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
    """Emit per-lane operand codes: IN_FEATURES weights then the bias, per lane."""
    for polarity, suffix in [('unipolar', 'u'), ('bipolar', 'b')]:
        lines = []
        for lane in range(LANES):
            for feature in range(IN_FEATURES):
                lines.append(WEIGHT_CODES[lane * IN_FEATURES + feature])
            lines.append(BIAS_CODES[lane])
        (VEC_DIR / f"linear_ugemm_operand_{suffix}.hex").write_text(
            "\n".join(f"{code:0{SEQ_WIDTH + 1}b}" for code in lines) + "\n"
        )


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["rst in_u_bits in_b_bits out_u out_u_nb out_b out_b_nb"]
    layer = run_sequence(rows)
    run_sequence(rows, dirty=TIMESTEPS // 2)
    assert rows[1:1 + TIMESTEPS] == rows[1 + TIMESTEPS:], \
        'reset() did not restore the opening sequence-index and accumulator state'

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
        f"`define GEN_PP_DELAY {pp_delay}\n"
        f"`define GEN_VECTORS {len(rows) - 1}\n"
    )
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} "
        f"(GEN_IN_FEATURES={IN_FEATURES}, GEN_LANES={LANES}, "
        f"GEN_SEQ_WIDTH={SEQ_WIDTH}, GEN_WIDTH={ACC_WIDTH}, "
        f"GEN_SCALE={IN_FEATURES + 1}, GEN_SCALE_NB={IN_FEATURES}, "
        f"GEN_PP_DELAY={pp_delay})"
    )


if __name__ == "__main__":
    main()
