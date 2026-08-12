"""
Generate golden test vectors for the sqrt_tracejkff RTL modules straight from
napl's functional Python model (napl.sim.operation.sqrt_tracejkff) -- so the
testbench checks the Verilog against the *actual* simulator, not a hand-derived
truth table.

sqrt_tracejkff is stateful (a JK-FF trace register and a decorr shuffle buffer
with its own position counter; bipolar also carries a width-2 bi2uni
accumulator). reset() clears all of them. We drive each polarity model
from reset() with the SAME input stream and record (in, out_unipolar,
out_bipolar) per cycle. The output at cycle t is what forward() returns at that
timestep, which is combinational in i_input given the cycle's incoming state.

Because the op is stateful, we also emit a MID-STREAM reset: after dirtying the
trace/acc registers with a leading segment, we call model.reset() and emit a
sentinel "R" row, then keep driving. The testbench pulses active-low i_rst_n on
the "R" row and must match the model again from the freshly-cleared state, which
proves the RTL reset is equivalent to the Python reset() from a dirty state.

Output: ../vec/sqrt_tracejkff.vec, one line per cycle:

    <in> <out_unipolar> <out_bipolar>   (each 0/1, space-separated)

or the sentinel reset row:

    R

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sqrt_tracejkff.py
"""
import math
import sys
from pathlib import Path

import torch
from napl.sim.operation import sqrt_tracejkff

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "sqrt_tracejkff.vec"
# decorr.v reads its position ROM from vec/decorr_rom.hex relative to the vvp
# cwd, which is this unit's directory, so the instantiated copy needs its own
# ROM here alongside the vectors.
ROM = VEC.parent / "decorr_rom.hex"

# The sqrt test encodes nonnegative inputs with a bipolar Sobol stream.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def clear_state(uni, bip):
    """Clear exactly the registers the RTL i_rst_n clears (trace, acc, shuffle buffer).

    Each sub-module is reset directly as well, so the cleared state is stated
    here rather than left to the base reset's traversal order.
    """
    uni.reset()
    bip.reset()
    uni.jkff.reset()
    bip.jkff.reset()
    bip.bi2uni.reset()
    uni.decorr.reset()
    bip.decorr.reset()


def write_rom(model):
    """Emit the shuffle-buffer position ROM, one word per timestep, as {idx_1, idx_0}."""
    seq_0, seq_1 = model.decorr.rand_seq_idx
    depth = model.decorr.depth
    idx_w = max(1, math.ceil(math.log2(depth)))
    lines = []
    for index_0, index_1 in zip(seq_0, seq_1):
        for index in (index_0, index_1):
            assert 0 <= index < depth, f"position {index} outside [0,{depth})"
        lines.append(f"{index_1:0{idx_w}b}{index_0:0{idx_w}b}")
    ROM.parent.mkdir(parents=True, exist_ok=True)
    with ROM.open("w") as rom:
        rom.write("\n".join(lines) + "\n")
    return len(lines)


def emit(f, uni, bip, bit):
    in_u = torch.tensor(bit, dtype=uni.stype)
    in_b = torch.tensor(bit, dtype=bip.stype)
    out_u = int(uni(in_u).item())
    out_b = int(bip(in_b).item())
    f.write(f"{bit} {out_u} {out_b}\n")


def main():
    uni = sqrt_tracejkff(config={"polarity": "unipolar"})
    bip = sqrt_tracejkff(config={"polarity": "bipolar"})

    # Both polarity models fix the same decorr configuration, so one ROM serves
    # both instantiated copies.
    assert uni.decorr.rand_seq_idx == bip.decorr.rand_seq_idx
    rom_words = write_rom(uni)

    values = rep_values(CODEC["polarity"], value_range=(0.0, 1.0))
    # Split where trace and accumulator state are nonzero.
    split = max(1, len(values) // 2)
    head_vals, tail_vals = values[:split], values[split:]

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        clear_state(uni, bip)
        for v in head_vals:
            for bit in encode_value(CODEC, v):
                emit(f, uni, bip, bit)
                rows += 1
        # R requests the corresponding active-low RTL reset.
        clear_state(uni, bip)
        f.write("R\n")
        rows += 1
        for v in tail_vals:
            for bit in encode_value(CODEC, v):
                emit(f, uni, bip, bit)
                rows += 1
    print(f"wrote {VEC} ({rows} vectors) and {ROM} ({rom_words} ROM words)")


if __name__ == "__main__":
    main()
