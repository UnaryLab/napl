"""
Generate golden test vectors for the min_rc RTL straight from napl's functional
Python model (napl.sim.operation.min_rc) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

min_rc is STATEFUL (holds dff and the sync_skewed counter), so a single static
input combination does not characterize it: we drive a multi-cycle stream from
model.reset() and record per-cycle (inputs, outputs). The testbench replays the
same stream after pulsing i_rst_n low, so both start from the post-reset() state.

To prove reset equivalence from a DIRTIED state, the stream is split into two
segments by a mid-stream model.reset(): the per-cycle `rst` flag marks the first
cycle of the second segment, where the testbench pulses i_rst_n low before
applying that cycle's inputs. The Python model is reset() at the same point, so
both sides re-enter the post-reset() state mid-stream and must agree afterward.

min_rc has NO config-derived sizing param: min_rc.__init__ calls
super().__init__(config, [], ...), and the sync_skewed width=2 it uses is an
intrinsic constant hardcoded in the model (not a user-tunable config size), so
there is no <op>_params.vh to emit.

Output: ../vec/min_rc.vec, one line per cycle:

    <in_0> <in_1> <min> <argmin> <rst>   (each 0/1, space-separated)

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_min_rc.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import min_rc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "min_rc.vec"

# Encoder settings mirror test_min_rc.py and use distinct Sobol dimensions.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def emit(f, model, s0, s1, rst_first):
    """Drive (s0, s1) through `model`, writing one vec line per cycle.

    `rst_first` marks the first cycle with rst=1 (the testbench pulses i_rst_n
    low before applying it); the caller is responsible for model.reset() so the
    Python state matches.
    """
    rows = 0
    for i, (a, b) in enumerate(zip(s0, s1)):
        o_min, o_arg = model(torch.tensor(a), torch.tensor(b))
        rst = 1 if (rst_first and i == 0) else 0
        f.write(f"{a} {b} {int(o_min.item())} {int(o_arg.item())} {rst}\n")
        rows += 1
    return rows


def main():
    model = min_rc(config={})

    s0, s1 = pair_streams(CODEC0, CODEC1, rep_pairs("bipolar", "bipolar"))

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        model.reset()
        rows += emit(f, model, s0, s1, rst_first=False)

        # The reset flag requests a matching RTL reset before replay.
        model.reset()
        rows += emit(f, model, s0, s1, rst_first=True)

    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()
