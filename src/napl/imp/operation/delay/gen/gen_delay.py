"""
Generate golden test vectors for the delay RTL module straight from napl's
functional Python model (napl.sim.operation.delay) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

delay is stateful: a depth-DEPTH bit-serial delay line (a FIFO of D flip-flops)
whose reset() state is the pattern config['init'] selects, all zeros under "zero"
and cell j holding j%2 under "alternate". Both init modes are driven here from the
same encoded spike stream, so one vec file covers the two hardware forms INIT
selects. We record (input, zero-init output, alternate-init output) every cycle.
The output at cycle t is what forward() returns at that timestep: the oldest cell,
read BEFORE the new input is pushed.

A MID-STREAM reset is injected to prove reset equivalence from a dirtied state:
after driving part of the stream, we call model.reset() (and emit a reset marker
the testbench replays as an i_rst_n pulse), then continue, so the co-sim checks
that RTL and model return to the identical initial pattern mid-run.

Output: ../vec/delay.vec, one line per cycle:

    <in> <out_zero> <out_alt>   (each 0/1, space-separated)
    R                           (a lone marker line: pulse i_rst_n low here)

The sizing param DEPTH is the single source of truth here: it is read from the
op config (mirroring test_delay.py's _DEPTH), used to build both models, AND
emitted into ../vec/delay_params.vh as `GEN_DEPTH so the testbench overrides the
RTL parameter with the same value. RTL and sim therefore inherit DEPTH from one
place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_delay.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import delay

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "delay.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "delay_params.vh"

# Sizing and encoder settings mirror test_delay.py.
DELAY = {"depth": 2}
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
# Column order of the per-cycle output fields, one model per init mode.
INITS = ("zero", "alternate")


def main():
    models = [delay(config=dict(DELAY, init=init)) for init in INITS]
    for model in models:
        model.reset()

    stream = []
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)

    # The midpoint marker tests reset after the FIFO has changed.
    reset_at = len(stream) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with VEC.open("w") as f:
        for idx, bit in enumerate(stream):
            if idx == reset_at:
                for model in models:
                    model.reset()
                f.write("R\n")  # testbench replays this as an i_rst_n pulse
            in_spike = torch.tensor(bit, dtype=models[0].stype)
            outs = [int(model(in_spike).item()) for model in models]
            f.write(f"{bit} {outs[0]} {outs[1]}\n")
            rows += 1

    pp_delay = models[0].hw.pp_delay
    PARAMS.write_text(
        f"`define GEN_DEPTH {DELAY['depth']}\n"
        f"`define GEN_PP_DELAY {pp_delay}\n"
        f"`define GEN_VECTORS {rows}\n"
    )
    print(
        f"wrote {VEC} ({rows} vectors, reset@{reset_at}) and {PARAMS} "
        f"(GEN_DEPTH={DELAY['depth']}, GEN_PP_DELAY={pp_delay})"
    )


if __name__ == "__main__":
    main()
