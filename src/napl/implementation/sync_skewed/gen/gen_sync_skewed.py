"""
Generate golden test vectors for the sync_skewed RTL module straight from napl's
functional Python model (napl.operation.sync_skewed) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

sync_skewed is stateful: a saturating WIDTH-bit counter buffers the lead/lag
between two spike streams. reset() sets cnt = 0. We drive the model from reset()
with a deterministic input stream chosen to exercise every regime (push, pop,
00/11 pass-through, and both saturation limits) and record (in_1, in_2, out_1,
out_2) every cycle. The output at cycle t is what forward() returns at that
timestep: computed from cnt BEFORE the update.

Output: ../vec/sync_skewed.vec, one line per cycle:

    <in_1> <in_2> <out_1> <out_2>     (each 0/1, space-separated)

The sizing param WIDTH is the single source of truth here: it is read from the
op config (mirroring test_sync_skewed.py's sync_skewed_config), used to build the
model, AND emitted into ../vec/sync_skewed_params.vh as `GEN_WIDTH so the
testbench overrides the RTL parameter with the same value. RTL and sim therefore
inherit WIDTH from one place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sync_skewed.py
"""
import sys
from pathlib import Path

import torch
from napl.operation import sync_skewed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "sync_skewed.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "sync_skewed_params.vh"

# test_sync_skewed.py sync_skewed_config: the sizing param the op is built with.
WIDTH = 3

# test_sync_skewed.py codec_config1/2: the two encoders feeding sync_skewed,
# both unipolar on distinct sobol dims (1 and 3). width=3 matches the test.
CODEC0 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 3}


def build_stream():
    # the test's encoder streams for representative unipolar operand pairs.
    s0, s1 = pair_streams(CODEC0, CODEC1, rep_pairs("unipolar", "unipolar"))
    return list(zip(s0, s1))


def emit_segment(model, f, stream):
    rows = 0
    for a, b in stream:
        in_1 = torch.tensor(a, dtype=model.stype)
        in_2 = torch.tensor(b, dtype=model.stype)
        o1, o2 = model(in_1, in_2)
        f.write(f"{a} {b} {int(o1.item())} {int(o2.item())}\n")
        rows += 1
    return rows


def main():
    model = sync_skewed(config={"width": WIDTH})
    model.reset()

    stream = build_stream()
    # Split the stream so we can re-issue reset() partway through (with the
    # counter dirtied), proving the RTL's async reset reproduces the model's
    # reset() from an arbitrary state. The "RST" marker drives the tb to pulse
    # i_rst_n low between segments.
    half = len(stream) // 2
    seg_a, seg_b = stream[:half], stream[half:]

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    PARAMS.write_text(f"`define GEN_WIDTH {WIDTH}\n")

    rows = 0
    with VEC.open("w") as f:
        rows += emit_segment(model, f, seg_a)
        # mid-stream reset: dirty cnt is cleared back to 0 in both model and RTL.
        f.write("RST\n")
        model.reset()
        rows += emit_segment(model, f, seg_b)
    print(f"wrote {VEC} ({rows} vectors, WIDTH={WIDTH}) and {PARAMS} "
          f"(GEN_WIDTH={WIDTH})")


if __name__ == "__main__":
    main()
