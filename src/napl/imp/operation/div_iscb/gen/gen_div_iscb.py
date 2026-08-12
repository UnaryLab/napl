"""
Generate golden test vectors for the div_iscb RTL modules straight from napl's
functional Python model (napl.sim.operation.div_iscb) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

div_iscb (in-stream correlation-based division) is stateful and has two polarity
variants. Each variant is its own model instance driven from reset() with the
SAME deterministic dividend/divisor spike streams; we record (dividend, divisor,
quotient) every cycle.

div_iscb's only config key is `polarity` (see test_div_iscb.py); the internal
widths/depths (sync_skewed width 3, div_cordiv depth 2, signabs/bi2uni/uni2bi
widths) are FIXED-OPTIMAL constants baked into the Python model, not config-
derived sizes -- so there is no sizing parameter to inherit and no params header
to emit (has_sizing_params = false). This generator validates the RTL only.

Because the op is stateful, the stream includes a MID-STREAM reset: at a chosen
cycle both models are reset() and a reset-marker row is emitted, so the testbench
re-pulses i_rst_n there and the co-sim proves the RTL returns to the model's
post-reset() state from a *dirtied* state, not just at t=0.

The bi2uni helper is also driven standalone, on its own stimulus column, because
its low clamp is unreachable through div_iscb_bipolar: signabs never holds its
magnitude output low for more than three cycles in a row, so the composed
helper's accumulator bottoms out at -3 and the clamp at -4 is never taken. Fed
directly, five consecutive silent cycles walk the accumulator from 0 to -4 and
take the clamp on the fifth, which is what makes the arm observable.

Output: ../vec/div_iscb.vec, one line per cycle:

    <reset> <dividend> <divisor> <quotient_unipolar> <quotient_bipolar> <b2u_in> <b2u_out>

`reset` is 1 on the marker row (the testbench pulses i_rst_n low for that cycle
and the model's pre-reset outputs on that row are don't-care / not checked), 0
otherwise. The two variants share the input columns so the testbench can drive
both DUTs from one stimulus, exactly like mul_gaines's vec format.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_div_iscb.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import bi2uni, div_iscb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "div_iscb.vec"

# Width the bipolar variant elaborates its bi2uni helpers at.
B2U_WIDTH = 3
# Six silent cycles then two spiking ones, appended after the main stream: the
# sixth silent cycle presents -5 to a clamp at -4.
B2U_TAIL = [0, 0, 0, 0, 0, 0, 1, 1]

# Inputs mirror test_div_iscb.py: distinct Sobol dimensions and
# |dividend| <= |divisor| with a nonzero divisor.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def build_streams():
    pairs = []
    for x, y in rep_pairs("bipolar", "bipolar"):
        lo, hi = (x, y) if abs(x) <= abs(y) else (y, x)
        hi = hi if hi != 0 else 1.0
        pairs.append((lo, hi))
    return pair_streams(CODEC0, CODEC1, pairs)


def main():
    uni = div_iscb(config={"polarity": "unipolar"})
    bi = div_iscb(config={"polarity": "bipolar"})
    b2u = bi2uni(config={"width": B2U_WIDTH})
    uni.reset()
    bi.reset()
    b2u.reset()

    dividend_stream, divisor_stream = build_streams()
    # The standalone helper reuses the dividend stream, then takes the tail.
    b2u_stream = list(dividend_stream) + B2U_TAIL
    dividend_stream = list(dividend_stream) + [0] * len(B2U_TAIL)
    divisor_stream = list(divisor_stream) + [1] * len(B2U_TAIL)
    total = len(dividend_stream)
    reset_at = total // 2
    clamped = 0

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for i, (dd, ds, bx) in enumerate(zip(dividend_stream, divisor_stream, b2u_stream)):
            if i == reset_at:
                # Reset rows carry don't-care outputs and reset every model before replay.
                f.write(f"1 {dd} {ds} x x {bx} x\n")
                rows += 1
                uni.reset()
                bi.reset()
                b2u.reset()
            t_dd = torch.tensor(dd, dtype=uni.stype)
            t_ds = torch.tensor(ds, dtype=uni.stype)
            out_uni = int(uni(t_dd, t_ds).item())
            out_bi = int(bi(t_dd, t_ds).item())
            # Count the cycles whose pre-clamp sum falls below the low bound.
            if int(b2u.accumulator.item()) + 2 * bx - 1 < b2u.acc_min:
                clamped += 1
            # bi2uni carries state across timesteps only while the input shape
            # matches its accumulator, so the helper is driven with a 1-element
            # tensor rather than a scalar one.
            out_b2u = int(b2u(torch.tensor([bx], dtype=b2u.stype)).item())
            f.write(f"0 {dd} {ds} {out_uni} {out_bi} {bx} {out_b2u}\n")
            rows += 1
    # Stimulus check: without a taken low clamp the arm is unobservable.
    assert clamped > 0, 'bi2uni low clamp was never taken'
    print(f"wrote {VEC} ({rows} vectors, mid-stream reset at row {reset_at}, "
          f"bi2uni low clamp taken on {clamped} cycle(s))")


if __name__ == "__main__":
    main()
