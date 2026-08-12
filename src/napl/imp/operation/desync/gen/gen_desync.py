"""
Generate golden test vectors for the desync RTL module straight from napl's
functional Python model (napl.sim.operation.desync) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

desync is stateful: a signed saved-one counter in [-DEPTH, DEPTH] paired with
the side that saves next. reset() sets cnt = 0 and side = +1. Both models are
built, one per polarity, and driven with the same stream: forward() reads no
polarity branch, so the two output streams must agree and one RTL module covers
both.

Output: ../vec/desync.vec, one line per cycle:

    <in_0> <in_1> <out_0> <out_1>     (each 0/1, space-separated)
    RST                               (pulse i_rst_n low here)

The sizing param DEPTH is the single source of truth here: it is read from the
op config (mirroring test_desync.py's make_operation), used to build the model,
AND emitted into ../vec/desync_params.vh as `GEN_DEPTH so the testbench
overrides the RTL parameter with the same value. RTL and sim therefore inherit
DEPTH from one place; they cannot drift. The header also records pp_delay and
the row count.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_desync.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import desync

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "desync.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "desync_params.vh"

# Sizing mirrors test_desync.py's make_operation.
DEPTH = 1

# Distinct Sobol dimensions decorrelate the two input streams.
CODEC0 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 3}


def build_stream():
    s0, s1 = pair_streams(CODEC0, CODEC1, rep_pairs("unipolar", "unipolar"))
    return list(zip(s0, s1))


def emit_segment(models, f, stream, seen):
    """Drive both polarity models over the stream and write one row per cycle."""
    rows = 0
    for a, b in stream:
        in_0 = torch.tensor(a, dtype=models[0].stype)
        in_1 = torch.tensor(b, dtype=models[0].stype)
        seen.add((int(models[0].cnt.item()), int(models[0].side.item())))
        outs = [model(in_0, in_1) for model in models]
        for other in outs[1:]:
            assert int(other[0].item()) == int(outs[0][0].item()), (a, b)
            assert int(other[1].item()) == int(outs[0][1].item()), (a, b)
        f.write(f"{a} {b} {int(outs[0][0].item())} {int(outs[0][1].item())}\n")
        rows += 1
    return rows


def main():
    models = [desync(config={"polarity": p, "depth": DEPTH})
              for p in ("unipolar", "bipolar")]
    for model in models:
        model.reset()

    stream = build_stream()
    # RST requests matching model and RTL resets between segments.
    half = len(stream) // 2
    seg_a, seg_b = stream[:half], stream[half:]

    VEC.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = 0
    with VEC.open("w") as f:
        rows += emit_segment(models, f, seg_a, seen)
        f.write("RST\n")
        for model in models:
            model.reset()
        rows += emit_segment(models, f, seg_b, seen)

    # Both save rails and both saving sides must be entered, or DEPTH and the
    # side flip would not be observable.
    assert (DEPTH, 1) in seen, seen
    assert (-DEPTH, -1) in seen, seen

    PARAMS.write_text(
        f"`define GEN_DEPTH {DEPTH}\n"
        f"`define GEN_PP_DELAY {models[0].hw.pp_delay}\n"
        f"`define GEN_VECTORS {rows}\n"
    )
    print(f"wrote {VEC} ({rows} vectors, DEPTH={DEPTH}, (cnt, side) states {sorted(seen)}) "
          f"and {PARAMS} (GEN_DEPTH={DEPTH}, GEN_PP_DELAY={models[0].hw.pp_delay})")


if __name__ == "__main__":
    main()
