"""
Generate golden test vectors for the jkff RTL module straight from napl's
functional Python model (napl.sim.operation.jkff) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/jkff.vec, one line per timestep:

    <input_j> <input_k> <out_q>      (each 0/1, space-separated)
    R                                (pulse i_rst_n low here)

jkff is stateful: its output depends on the held state q (reset to 0). A single
multi-cycle stream that visits both q states under every (j, k) input fully
characterizes the transition table. The stream also resets after changing q, so
the co-sim checks recovery from a changed state as well as the opening state.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_jkff.py
"""
from pathlib import Path

import torch
from napl.sim.operation import jkff

VEC = Path(__file__).resolve().parent.parent / "vec" / "jkff.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "jkff_params.vh"

# The prefix matches test_jkff.py's raw inputs; the suffix covers both q states.
STREAM = [
    (0, 0), (0, 1), (1, 0), (1, 1),
    (1, 1), (1, 0), (0, 1), (0, 0),
    (0, 0),
    (1, 0),
    (0, 0),
    (0, 1),
    (1, 1),
    (1, 1),
    (1, 0),
    (1, 1),
    (0, 1),
    (1, 0),
    (0, 0),
    (0, 1),  # q:1->0
]


def main():
    model = jkff(config={})
    model.reset()
    reset_at = len(STREAM) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_PP_DELAY {model.hw.pp_delay}\n")
    rows = 0
    with VEC.open("w") as f:
        for idx, (j, k) in enumerate(STREAM):
            if idx == reset_at:
                model.reset()
                f.write("R\n")
            in_j, in_k = torch.tensor(j), torch.tensor(k)
            out_q = int(model(in_j, in_k).item())
            f.write(f"{j} {k} {out_q}\n")
            rows += 1
    print(
        f"wrote {VEC} ({rows} vectors, reset@{reset_at}) and {PARAMS} "
        f"(GEN_PP_DELAY={model.hw.pp_delay})"
    )


if __name__ == "__main__":
    main()
