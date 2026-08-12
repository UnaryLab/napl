"""
Generate golden test vectors for the sqrt_traceiscb RTL modules straight from
napl's functional Python model (napl.sim.operation.sqrt_traceiscb) -- so the
testbenches check the Verilog against the *actual* simulator, not a hand-derived
truth table.

sqrt_traceiscb is a stateful bit-serial square-root circuit (stochastic bit
inserting via the in-stream correlation-based division cordiv kernel). It has two
polarity variants (unipolar / bipolar) with independent state, so we drive BOTH
models from reset() with the same deterministic 0/1 stream and record per cycle:

    ../vec/sqrt_traceiscb.vec, one line per cycle:

        <rst> <in> <out_unipolar> <out_bipolar>   (each 0/1, space-separated)

The output at cycle t is what forward() returns at that timestep. <rst> is a
reset marker: when 1, the cycle applies the model's reset() BEFORE driving the
input, and the recorded outputs are those of the freshly-reset model. We inject a
mid-stream reset partway through (after the streams have dirtied every register)
so the co-sim proves the RTL's active-low i_rst_n reproduces reset() from an
arbitrary dirtied state, not just at t=0.

The encoded stream is followed by a block of short words each driven from its
own reset, which pin the post-reset cordiv buffer and shuffle-buffer values.
assert_model_matches_rtl() requires step() to reproduce, cycle by cycle, the
state trajectory the compiled RTL prints under tb/sqrt_traceiscb_probe.v, an
exhaustive sweep of all 4096 12-bit input words from reset. The reference
therefore comes from the Verilog, not from this file's own recurrence, so a
model edit that changes the recurrence fails generation.

The trace path carries a decorr shuffle buffer whose position sequence has a
period of 256, so the state reachable from reset is no longer enumerable inside
a 12-cycle probe word and the transition-set coverage oracle this file once ran
has been retired with it. The golden-vector co-simulation and the trajectory
check above are what pin the RTL.

sqrt_traceiscb has no config-derived sizing param: its only config key is
'polarity'. The cordiv depth (2), the bi2uni width (3), and the shuffle buffer's
depth (4) and period (256) are intrinsic algorithm constants fixed inside the
op's __init__ ("the config is fixed to optimal directly"), not derived from the
op's own config -- so no <op>_params.vh is emitted and the RTL carries no
parameter. The shuffle buffer's position ROM is emitted here as
vec/decorr_rom.hex, which the instantiated decorr copy reads relative to the
unit directory.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sqrt_traceiscb.py
"""
import math
import subprocess
import sys
from itertools import product
from pathlib import Path

import torch
from napl.sim.operation import sqrt_traceiscb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

UNIT = Path(__file__).resolve().parent.parent
VEC = UNIT / "vec" / "sqrt_traceiscb.vec"
# decorr.v reads its position ROM from vec/decorr_rom.hex relative to the vvp
# cwd, which is this unit's directory, so the instantiated copy needs its own
# ROM here alongside the vectors.
ROM = UNIT / "vec" / "decorr_rom.hex"
PROBE = UNIT / "tb" / "sqrt_traceiscb_probe.v"
PROBE_WORDS = 4096
PROBE_LEN = 12

# The test's own encoder settings drive the full bipolar [-1, 1] range, wider
# than the unipolar [0, 1] operands test_sqrt_traceiscb.py draws, so stimulus
# reaches a branch no test row enters while every stream still comes from the
# real napl encoder.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def from_reset_words():
    """Short input words each driven from a fresh reset.

    The length-6 sweep walks the cordiv and trace registers over every input
    pattern of that length, and the six leading zeros of the second family walk
    the bipolar accumulator down to its lower clamp before the same sweep
    resumes. Driving each word from reset also pins the cordiv buffer's and the
    shuffle buffer's reset values, which are observable only in the cycles right
    after a reset. The whole block costs about 7 ms of simulation.
    """
    short = [list(w) for w in product((0, 1), repeat=6)]
    return short + [[0] * 6 + w for w in short]


def rows_from_stimulus():
    """(reset_marker, input_bit) rows for the whole vector file.

    The encoded block runs one long stream with a midpoint reset, checking the
    post-reset trajectory from a dirtied state; the from-reset words follow,
    each marking a reset on its first cycle.
    """
    stream = []
    for v in rep_values(CODEC["polarity"], value_range=(-1.0, 1.0)):
        stream += encode_value(CODEC, v)
    reset_at = len(stream) // 2
    rows = [(1 if i == reset_at else 0, bit) for i, bit in enumerate(stream)]
    for word in from_reset_words():
        rows += [(1 if i == 0 else 0, bit) for i, bit in enumerate(word)]
    return rows, reset_at


# Reset state of the RTL registers: (trace, dff, buf0, buf1, idx), the bipolar
# arm's bi2uni accumulator, then the shuffle buffer's cells (cell s resets to
# s % 2, position DEPTH-1 is the pass-through path holding no cell) and position
# counter.
SHUFFLE_DEPTH = 4
SHUFFLE_CELLS = tuple(slot % 2 for slot in range(SHUFFLE_DEPTH - 1))
RESET_STATE = {False: (0, 0, 0, 1, 0, SHUFFLE_CELLS, 0),
               True: (0, 0, 0, 1, 0, 0, SHUFFLE_CELLS, 0)}


def step(state, bit, bipolar, positions):
    """One clock edge of the RTL recurrence, returning the next state."""
    trace, dff, buf0, buf1, idx = state[:5]
    cells, seq = state[-2:]
    output_bit = trace | bit
    acc_tail = ()
    if bipolar:
        acc = min(max(state[5] + (1 if output_bit else -1), -4), 3)
        out_bit = 1 if acc >= 1 else 0
        acc_tail = (acc - out_bit,)
    else:
        out_bit = output_bit
    # Shuffle buffer: the selected position either passes the current bit
    # through or emits the bit stored there and stores the current one.
    position = positions[seq]
    if position == SHUFFLE_DEPTH - 1:
        shuffled, cells_next = out_bit, cells
    else:
        shuffled = cells[position]
        cells_next = cells[:position] + (out_bit,) + cells[position + 1:]
    dividend = (1 - dff) & shuffled
    divisor = dff | dividend
    quotient = dividend if divisor else (buf0 if idx == 0 else buf1)
    nxt = (quotient, 1 - dff,
           buf1 if divisor else buf0,
           quotient if divisor else buf1,
           1 - idx)
    return nxt + acc_tail + (cells_next, (seq + 1) % len(positions))


def write_rom(model):
    """Emit the shuffle-buffer position ROM, one word per timestep, as {idx_1, idx_0}."""
    seq_0, seq_1 = model.decorr.rand_seq_idx
    depth = model.decorr.depth
    assert depth == SHUFFLE_DEPTH, depth
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


def probe_cycles():
    """Per-cycle (unipolar state, bipolar state, input bit) read off the RTL.

    Compiles rtl/*.v with tb/sqrt_traceiscb_probe.v and runs it: the probe drives
    both variants over all 4096 12-bit words, each from its own reset, and prints
    the register state standing before every posedge together with the input bit.
    Nothing in this file contributes to the result, so it is an independent
    reference rather than an echo of step().
    """
    build = UNIT / "build"
    build.mkdir(parents=True, exist_ok=True)
    exe = build / "probe"
    sources = sorted(str(p) for p in (UNIT / "rtl").glob("*.v"))
    # The variants instantiate decorr, so its rtl/ joins the library search path
    # the Makefile builds for the unit testbench.
    subprocess.run(["iverilog", "-g2001", "-Wall",
                    "-y", str(UNIT.parent / "decorr" / "rtl"),
                    "-o", str(exe), *sources, str(PROBE)],
                   check=True)
    # decorr resolves its position ROM against the working directory, which is
    # the unit directory under both `make test` and this generator.
    out = subprocess.run(["vvp", str(exe)], check=True, cwd=str(UNIT),
                         capture_output=True, text=True).stdout
    lines = out.strip().splitlines()
    assert len(lines) == PROBE_WORDS * PROBE_LEN, (
        f'probe printed {len(lines)} cycles, expected {PROBE_WORDS * PROBE_LEN}')
    cycles = []
    for line in lines:
        uni, bi, acc, uni_cells, bi_cells, seq, bit = line.split()
        # acc_q is a signed 4-bit register printed as raw bits.
        acc_val = int(acc, 2) - (16 if acc[0] == "1" else 0)
        # store_q[DEPTH-2:0] prints most significant cell first.
        uni_tail = (tuple(int(c) for c in reversed(uni_cells)), int(seq))
        bi_tail = (tuple(int(c) for c in reversed(bi_cells)), int(seq))
        cycles.append((tuple(int(c) for c in uni) + uni_tail,
                       tuple(int(c) for c in bi) + (acc_val,) + bi_tail,
                       int(bit)))
    return cycles


def assert_model_matches_rtl(cycles, positions):
    """Require step() to reproduce the probe's own state trajectory, cycle by cycle.

    The probe resets before each 12-bit word, so the comparison starts from the
    reset state 4096 times per arm and covers every register the RTL carries,
    including the accumulator clamp arms and the shuffle buffer's cells. A model
    edit that changes the recurrence, and an RTL edit that changes the hardware,
    part here on the first cycle they disagree.
    """
    checked = []
    for bipolar in (False, True):
        state = RESET_STATE[bipolar]
        for i, cycle in enumerate(cycles):
            if i % PROBE_LEN == 0:
                state = RESET_STATE[bipolar]  # the probe resets before each word
            observed, bit = cycle[1 if bipolar else 0], cycle[2]
            arm = 'bipolar' if bipolar else 'unipolar'
            assert state == observed, (
                f'the {arm} model state {state} parts from the RTL state {observed} '
                f'at probe cycle {i} (word {i // PROBE_LEN}, input {bit})')
            state = step(state, bit, bipolar, positions)
        checked.append(len(cycles))
    return checked


def main():
    uni = sqrt_traceiscb(config={"polarity": "unipolar"})
    bi = sqrt_traceiscb(config={"polarity": "bipolar"})
    uni.reset()
    bi.reset()

    # Both polarity models fix the same decorr configuration, so one ROM serves
    # both instantiated copies.
    assert uni.decorr.rand_seq_idx == bi.decorr.rand_seq_idx
    rom_words = write_rom(uni)
    positions = uni.decorr.rand_seq_idx[0]

    rows, reset_at = rows_from_stimulus()
    # The ROM must exist before the probe elaborates, since decorr reads it at
    # time zero; probe_cycles() runs vvp from this unit's directory.
    uni_cycles, bi_cycles = assert_model_matches_rtl(probe_cycles(), positions)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    with VEC.open("w") as f:
        for do_reset, bit in rows:
            if do_reset:
                uni.reset()
                bi.reset()
            inp_u = torch.tensor(bit, dtype=uni.stype)
            inp_b = torch.tensor(bit, dtype=bi.stype)
            out_uni = int(uni(inp_u).item())
            out_bi = int(bi(inp_b).item())
            f.write(f"{do_reset} {bit} {out_uni} {out_bi}\n")
    print(f"wrote {VEC} ({len(rows)} vectors, mid-stream reset at cycle {reset_at}) "
          f"and {ROM} ({rom_words} ROM words); step() matched the RTL probe over "
          f"{uni_cycles} unipolar and {bi_cycles} bipolar cycles")


if __name__ == "__main__":
    main()
