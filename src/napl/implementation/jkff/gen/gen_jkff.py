"""
Generate golden test vectors for the jkff RTL module straight from napl's
functional Python model (napl.operation.jkff) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/jkff.vec, one line per timestep:

    <input_j> <input_k> <out_q>      (each 0/1, space-separated)

jkff is stateful: its output depends on the held state q (reset to 0). A single
multi-cycle stream that visits both q states under every (j, k) input fully
characterizes the transition table. The model is driven from reset() at t=0, one
forward() call per timestep, recording (inputs, output) each cycle; the testbench
pulses i_rst_n low before replaying, so the co-sim starts from the same t=0 state.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_jkff.py
"""
from pathlib import Path

import torch
from napl.operation import jkff

VEC = Path(__file__).resolve().parent.parent / "vec" / "jkff.vec"

# test_jkff.py feeds the op raw (j, k) spike vectors directly (no encoder), so the
# faithful "same input as the test" is those exact j/k sequences. The test drives
# j=[0,0,1,1] k=[0,1,0,1] then j=[1,1,0,0] k=[1,0,1,0]; we lead with those, then
# continue with a stream that visits both q states under every (j, k) input so the
# full transition table is also exercised.
STREAM = [
    # test_jkff.py sequence 1: (j,k) zipped from [0,0,1,1] / [0,1,0,1]
    (0, 0), (0, 1), (1, 0), (1, 1),
    # test_jkff.py sequence 2: (j,k) zipped from [1,1,0,0] / [1,0,1,0]
    (1, 1), (1, 0), (0, 1), (0, 0),
    # exhaustive transition coverage across both q states.
    (0, 0),  # q:0->0  (J=0)
    (1, 0),  # q:0->1  (set)
    (0, 0),  # q:1->1  (K=0)
    (0, 1),  # q:1->0  (reset)
    (1, 1),  # q:0->1  (toggle from 0)
    (1, 1),  # q:1->0  (toggle from 1)
    (1, 0),  # q:0->1
    (1, 1),  # q:1->0
    (0, 1),  # q:0->0
    (1, 0),  # q:0->1
    (0, 0),  # q:1->1
    (0, 1),  # q:1->0
]


def main():
    model = jkff(config={})
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for j, k in STREAM:
            in_j, in_k = torch.tensor(j), torch.tensor(k)
            out_q = int(model(in_j, in_k).item())
            f.write(f"{j} {k} {out_q}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()
