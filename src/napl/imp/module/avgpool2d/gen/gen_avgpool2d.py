"""
Generate golden test vectors for the avgpool2d module RTL straight from napl's
functional Python model (napl.sim.module.avgpool2d) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

avgpool2d is stateful: each pooled output position keeps a residual accumulator
that adds the window mean every timestep and emits a spike once the residual
reaches one. The RTL keeps the same residual as an integer in units of
1/KERNEL_AREA, which is bit-exact with the model while 1/KERNEL_AREA is
representable in the model dtype (the verified configurations use power-of-two
window areas).

The vectors cover unpadded windows only.

Three sequences follow each other: a unipolar stream, a bipolar stream, and the
unipolar stream replayed after half a stream of bipolar spikes has dirtied the
accumulators and reset() has cleared them, which must reproduce the first
sequence bit for bit.

Output: ../vec/avgpool2d.vec, one line per timestep:

    <rst> <in_bits> <out_bits>

`in_bits` is LANES*KERNEL_AREA binary digits, MSB first, so lane l occupies
i_input_spike[l*KERNEL_AREA +: KERNEL_AREA] with the window flattened row-major.
`out_bits` is LANES binary digits, MSB first, in the same lane order (the pooled
output flattened row-major). `rst` is 1 on the first timestep of each independent
sequence, where the testbench pulses i_rst_n low.

Sizing values come from this file only and are emitted into
../vec/avgpool2d_params.vh, so the testbench elaborates the RTL at the model's
configuration and the two cannot drift. The header also records pp_delay.

check_mapping() translates this module's own mapping.yaml entry and requires the
resolved parameters to equal the ones used here, so the co-simulation gates the
mapping entry as well as the RTL.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_avgpool2d.py
"""
import math
import sys
from pathlib import Path

import torch

from napl.sim.base import global_config
from napl.sim.module import avgpool2d
from napl.sim.operation import encode
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "operation"))
from _gen_common import require_seeded_sys  # noqa: E402

VEC = Path(__file__).resolve().parent.parent / "vec" / "avgpool2d.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "avgpool2d_params.vh"

# Geometry, input shape, and codec mirror tests/module/test_avgpool2d.py.
KERNEL_SIZE = 2
SHAPE = (4, 3, 8, 8)
TIMESTEPS = 256

KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE
LANES = math.prod(SHAPE) // KERNEL_AREA


def check_mapping():
    """Check mapping.yaml resolves this module's own sizing parameters.

    The vectors above are built straight from the Python model, so without this
    the mapping entry could drift from the hardware the co-simulation verifies.
    """
    node = {"class": "avgpool2d", "config": {"kernel_size": KERNEL_SIZE, "lanes": LANES}}
    assert translate_node(node).parameters == {
        "KERNEL_AREA": KERNEL_AREA, "LANES": LANES}, \
        f"mapping.yaml avgpool2d resolves {translate_node(node).parameters}"


def window_bits(spike):
    """Flatten a spike tensor into per-lane windows, lane-major then row-major."""
    batch, channel, height, width = spike.shape
    windows = spike.reshape(batch, channel, height // KERNEL_SIZE, KERNEL_SIZE,
                            width // KERNEL_SIZE, KERNEL_SIZE)
    return windows.permute(0, 1, 2, 4, 3, 5).reshape(-1).to(torch.int64).tolist()


def as_binary(bits):
    """Render a bit list as a Verilog %b string, highest bit index first."""
    return "".join(str(int(bit)) for bit in reversed(bits))


# Bipolar values run over a narrower range than unipolar ones, so the encoded
# bipolar rate (x + 1) / 2 differs from the unipolar rate x.
RANGE = {'unipolar': (0.0, 1.0), 'bipolar': (-1.0, 0.75)}


def values(polarity):
    """The test's fidelity-scale input tensor for one polarity."""
    low, high = RANGE[polarity]
    return torch.linspace(low, high, math.prod(SHAPE),
                          dtype=global_config.ntype).reshape(SHAPE)


def run_sequence(polarity, rows, dirty=0):
    """Append one reset-to-reset sequence of vectors for the given polarity.

    `dirty` timesteps of the bipolar stream run before the recorded sequence and
    are followed by reset(), so the recorded half starts from accumulators that
    held state. Its outputs must equal the same sequence recorded from a fresh
    model, which the testbench checks by pulsing i_rst_n on the opening rst=1 row.
    """
    codec = {'polarity': polarity, 'timestep': TIMESTEPS, 'generator': 'sobol', 'dim': 1}
    require_seeded_sys(codec)
    enc = encode(codec)
    enc.reset()
    pool = avgpool2d(KERNEL_SIZE, config={'polarity': polarity})
    pool.reset()

    if dirty:
        dirty_codec = {'polarity': 'bipolar', 'timestep': TIMESTEPS,
                       'generator': 'sobol', 'dim': 1}
        require_seeded_sys(dirty_codec)
        dirty_enc = encode(dirty_codec)
        dirty_enc.reset()
        dirty_value = values('bipolar')
        for _ in range(dirty):
            pool(dirty_enc(dirty_value))
        pool.reset()

    value = values(polarity)
    for timestep in range(TIMESTEPS):
        spike = enc(value)
        out = pool(spike)
        rows.append(
            f"{1 if timestep == 0 else 0} "
            f"{as_binary(window_bits(spike))} "
            f"{as_binary(out.reshape(-1).to(torch.int64).tolist())}"
        )
    return pool.hw.pp_delay


def main():
    check_mapping()
    rows = ["rst in_bits out_bits"]
    # Both polarities drive the same circuit: without padding the pooled window is
    # polarity-independent, so the two sequences differ only in their spike stream.
    pp_delay = 0
    for polarity in ['unipolar', 'bipolar']:
        pp_delay = run_sequence(polarity, rows)
    # Replay the unipolar sequence after a mid-stream reset of dirtied accumulators.
    run_sequence('unipolar', rows, dirty=TIMESTEPS // 2)
    assert rows[1:1 + TIMESTEPS] == rows[1 + 2 * TIMESTEPS:], \
        'reset() did not restore the opening accumulator state'
    # Bipolar p = (x + 1) / 2 over the unipolar grid would reproduce the unipolar
    # block exactly, leaving the bipolar block no distinct stimulus.
    assert rows[1:1 + TIMESTEPS] != rows[1 + TIMESTEPS:1 + 2 * TIMESTEPS], \
        'bipolar stimulus is identical to unipolar'

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_KERNEL_AREA {KERNEL_AREA}\n"
        f"`define GEN_LANES {LANES}\n"
        f"`define GEN_PP_DELAY {pp_delay}\n"
        f"`define GEN_VECTORS {len(rows) - 1}\n"
    )
    VEC.write_text("\n".join(rows) + "\n")
    print(
        f"wrote {VEC} ({len(rows) - 1} vectors) and {PARAMS} "
        f"(GEN_KERNEL_AREA={KERNEL_AREA}, GEN_LANES={LANES}, "
        f"GEN_PP_DELAY={pp_delay})"
    )


if __name__ == "__main__":
    main()
