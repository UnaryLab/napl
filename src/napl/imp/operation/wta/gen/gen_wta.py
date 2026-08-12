"""
Generate golden vectors for the wta RTL module from napl's Python model
(napl.sim.operation.wta), so the testbench checks the Verilog against the actual
simulator rather than a hand-derived truth table.

Output: ../vec/wta.vec, one line per timestep:

    <rst> <input> <output>

`input` is the ENTRY stacked streams packed as a Verilog vector, stream 0 in the
least significant bit; `rst` and `output` are single bits.

The stimulus has two parts. The edge segments drive the falling-edge cases
directly: no edge at all, an early edge that suppresses a later one, and
simultaneous earliest edges. The fidelity segments then encode the three-stream
operand triples of test_wta.py with its temporal encoder, in both polarities the
test sweeps. Each segment is an independent run, so every segment starts with a
reset.

Run inside the `napl` conda env:
    python gen/gen_wta.py
"""
import sys
from pathlib import Path

import torch

from napl.sim.operation import wta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "wta.vec"
PARAMS = ROOT / "vec" / "wta_params.vh"

TIMESTEP = 256
# test_wta.py stacks three streams and encodes each with the temporal generator.
ENTRY = 3
POLARITIES = ("unipolar", "bipolar")
# Every fourth value of test_wta.py's 128-point sweep, with its two rolls, which
# keeps the operand triples of the test while holding the vector file small.
STRIDE = 4
EDGE_TIMESTEPS = 6
# Falling times per stacked stream; None never falls.
EDGE_CASES = [
    (None, None, None),   # no edge at all
    (4, 2, None),         # the earliest edge wins, the later one stays suppressed
    (2, 2, 2),            # simultaneous earliest edges merge into one spike
    (0, 3, 5),            # an edge on the opening timestep
]


def codec(polarity, dim):
    """The encoder configuration test_wta.py builds for one stacked stream."""
    return {"polarity": polarity, "timestep": TIMESTEP, "generator": "temporal",
            "dim": dim}


def edge_segments():
    """Directly driven stacked streams for the falling-edge cases."""
    segments = []
    for falls in EDGE_CASES:
        segments.append([
            [1 if fall is None or timestep < fall else 0 for fall in falls]
            for timestep in range(EDGE_TIMESTEPS)
        ])
    return segments


def fidelity_segments():
    """Encoded stacked streams over test_wta.py's operand triples."""
    segments = []
    for polarity in POLARITIES:
        low = 0.0 if polarity == "unipolar" else -1.0
        values = torch.linspace(low, 1.0, 128)
        triples = zip(values[::STRIDE].tolist(),
                      values.roll(31)[::STRIDE].tolist(),
                      values.roll(67)[::STRIDE].tolist())
        for triple in triples:
            streams = [encode_value(codec(polarity, dim + 1), value)
                       for dim, value in enumerate(triple)]
            segments.append([list(spikes) for spikes in zip(*streams)])
    return segments


def bus(spikes):
    """Format stream 0 as the Verilog vector's least significant bit."""
    return "".join(str(int(bit)) for bit in reversed(spikes))


def main():
    model = wta()
    segments = edge_segments() + fidelity_segments()

    rows = []
    for segment in segments:
        model.reset()
        for index, spikes in enumerate(segment):
            assert len(spikes) == ENTRY
            result = int(model(torch.tensor(spikes, dtype=model.stype), dim=-1).item())
            rows.append((int(index == 0), bus(spikes), result))
    # A winner spike and a suppressed later edge are the two behaviors the state
    # holds, so a stimulus reaching neither would pass on the silent rows alone.
    assert any(row[2] for row in rows), "no segment produced a winner spike"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )
    with VEC.open("w") as output:
        for reset, spikes, result in rows:
            output.write(f"{reset} {spikes} {result}\n")

    print(f"wrote {VEC} ({len(rows)} vectors, {len(segments)} reset segments) and "
          f"{PARAMS} (GEN_ENTRY={ENTRY} GEN_PP_DELAY={model.hw.pp_delay})")


if __name__ == "__main__":
    main()
