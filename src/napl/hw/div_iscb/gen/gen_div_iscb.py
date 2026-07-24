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

Output: ../vec/div_iscb.vec, one line per cycle:

    <reset> <dividend> <divisor> <quotient_unipolar> <quotient_bipolar>

`reset` is 1 on the marker row (the testbench pulses i_rst_n low for that cycle
and the model's pre-reset outputs on that row are don't-care / not checked), 0
otherwise. The two variants share the input columns so the testbench can drive
both DUTs from one stimulus, exactly like mul_and's vec format.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_div_iscb.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import div_iscb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "div_iscb.vec"

# test_div_iscb.py codec_config1/2: both bipolar on distinct sobol dims (1, 2).
# The test sorts so |dividend| <= |divisor| (proper-fraction quotient) and the
# divisor != 0.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def build_streams():
    # representative pairs with |dividend| <= |divisor| (!=0), mirroring the test.
    pairs = []
    for x, y in rep_pairs("bipolar", "bipolar"):
        lo, hi = (x, y) if abs(x) <= abs(y) else (y, x)
        hi = hi if hi != 0 else 1.0
        pairs.append((lo, hi))
    return pair_streams(CODEC0, CODEC1, pairs)


def main():
    uni = div_iscb(config={"polarity": "unipolar"})
    bi = div_iscb(config={"polarity": "bipolar"})
    uni.reset()
    bi.reset()

    dividend_stream, divisor_stream = build_streams()
    total = len(dividend_stream)
    # mid-stream reset point: after the models are well dirtied, before the end.
    reset_at = total // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for i, (dd, ds) in enumerate(zip(dividend_stream, divisor_stream)):
            if i == reset_at:
                # Emit a reset-marker row: TB pulses i_rst_n low; outputs on this
                # row are don't-care (1'bx) so they are not checked. Then reset
                # the models so subsequent rows compare from the post-reset state.
                f.write(f"1 {dd} {ds} x x\n")
                rows += 1
                uni.reset()
                bi.reset()
            t_dd = torch.tensor(dd, dtype=uni.stype)
            t_ds = torch.tensor(ds, dtype=uni.stype)
            out_uni = int(uni(t_dd, t_ds).item())
            out_bi = int(bi(t_dd, t_ds).item())
            f.write(f"0 {dd} {ds} {out_uni} {out_bi}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors, mid-stream reset at row {reset_at})")


if __name__ == "__main__":
    main()
