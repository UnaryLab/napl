"""
Generate golden test vectors for the eq_rc RTL module straight from napl's
functional Python model (napl.sim.operation.eq_rc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

eq_rc is stateful: it holds a running signed spike-count difference `count` and,
via napl_base, an elapsed-timestep counter `t`, both reset to 0. The per-timestep
output is combinational on the post-update state:

    y_t = 1{ value_gain*|count| / t <= tolerance }
        = 1{ value_gain*TOL_DEN*|count| <= TOL_NUM*t },

where value_gain is 1 (unipolar) or 2 (bipolar) and tolerance = TOL_NUM/TOL_DEN.
The value_gain is the only polarity difference, so one parameterized module covers
both polarities; the two polarities are driven as two lockstep streams (their
encoders differ, so each polarity carries its own input pair).

Output: ../vec/eq_rc.vec, one line per cycle:

    <rst_n> <in0_u> <in1_u> <out_u> <in0_b> <in1_b> <out_b>   (each 0/1)

A line with rst_n==0 is a reset pulse shared by both instances: the inputs/outputs
are don't-care (0) and the TB asserts i_rst_n low for that cycle, proving the
active-low reset returns count->0 / t->0 from a dirtied state (matching reset()).

Also emits ../vec/eq_rc_params.vh with GEN_TOL_NUM / GEN_TOL_DEN (the tolerance as
an exact fraction) and GEN_TW (the elapsed-timestep counter width sized to the
driven stream), the single source of truth the RTL and TB inherit from.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_eq_rc.py
"""
import sys
from fractions import Fraction
from pathlib import Path

import torch
from napl.sim.operation import eq_rc
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import pair_streams, rep_pairs

VECDIR = Path(__file__).resolve().parent.parent / "vec"
VEC = VECDIR / "eq_rc.vec"
HDR = VECDIR / "eq_rc_params.vh"

# Rate-equality band, mirroring _TOLERANCE in tests/operation/test_eq_rc.py.
TOLERANCE = 0.1
# Encoder timestep and sobol dims mirror the streaming test's codec config.
TIMESTEP = 256


def codecs(polarity):
    c0 = {"polarity": polarity, "timestep": TIMESTEP, "generator": "sobol", "dim": 1}
    c1 = {"polarity": polarity, "timestep": TIMESTEP, "generator": "sobol", "dim": 2}
    return c0, c1


def build_rows(polarity, pairs):
    """Per-cycle (rst_n, in0, in1, out) rows; each pair is its own reset segment.

    Each representative value pair is encoded and run as a fresh TIMESTEP-long
    segment, with the model reset (and a reset-pulse row emitted) between pairs.
    No run then exceeds TIMESTEP, so the verified elapsed-timestep counter is
    sized to a single deployment run rather than a concatenation of many.

    Returns (rows, max_t) with max_t the largest elapsed timestep reached, so the
    counter width matches the per-run deployment sizing (== TIMESTEP).
    """
    model = eq_rc({"polarity": polarity, "tolerance": TOLERANCE})
    c0, c1 = codecs(polarity)
    rows, max_t = [], 0
    for pair_index, pair in enumerate(pairs):
        # Reset count=0 / t=0 before each segment so no run exceeds TIMESTEP.
        model.reset()
        if pair_index > 0:
            rows.append((0, 0, 0, 0))
        s0, s1 = pair_streams(c0, c1, [pair])
        for a, b in zip(s0, s1):
            out = int(model(torch.tensor(a), torch.tensor(b)).item())
            rows.append((1, a, b, out))
            max_t = max(max_t, model.timestep_cur)
    return rows, max_t


def check_mapping(num, den, tw):
    """Fail if mapping.yaml's eq_rc params drift from this gen's elaborated values.

    The testbench elaborates two DUTs (VALUE_GAIN 1 and 2) at TOL_NUM/TOL_DEN/TW
    fixed by this gen. Resolving the mapping entry for a natural deployment node
    (timestep == TIMESTEP, so len(SEGMENT) == the verified run length) and
    requiring it to reproduce those exact params ties the shipped config to the
    co-sim-verified one, so a mapping formula that drifts from the verified width
    trips this assert during `make test`.
    """
    for polarity in ("unipolar", "bipolar"):
        binding = translate_node({
            "class": "eq_rc",
            "config": {"polarity": polarity, "tolerance": TOLERANCE, "timestep": TIMESTEP},
        })
        expected = {
            "VALUE_GAIN": 2 if polarity == "bipolar" else 1,
            "TOL_NUM": num,
            "TOL_DEN": den,
            "TW": tw,
        }
        assert binding.parameters == expected, \
            f"mapping.yaml eq_rc ({polarity}) resolves {binding.parameters}, not {expected}"


def main():
    pairs_u = rep_pairs("unipolar", "unipolar")
    pairs_b = rep_pairs("bipolar", "bipolar")
    # Both instances share one clock/reset, so the two streams must run lockstep.
    m = min(len(pairs_u), len(pairs_b))
    pairs_u, pairs_b = pairs_u[:m], pairs_b[:m]

    rows_u, tu = build_rows("unipolar", pairs_u)
    rows_b, tb = build_rows("bipolar", pairs_b)
    assert len(rows_u) == len(rows_b), (len(rows_u), len(rows_b))

    max_t = max(tu, tb)
    tw = max(1, max_t).bit_length() + 1

    frac = Fraction(TOLERANCE).limit_denominator(1000000)
    num, den = frac.numerator, frac.denominator

    check_mapping(num, den, tw)

    VECDIR.mkdir(parents=True, exist_ok=True)
    with VEC.open("w") as f:
        for (ru, a_u, b_u, o_u), (rb, a_b, b_b, o_b) in zip(rows_u, rows_b):
            assert ru == rb, (ru, rb)
            f.write(f"{ru} {a_u} {b_u} {o_u} {a_b} {b_b} {o_b}\n")
    with HDR.open("w") as f:
        f.write("// Generated by gen_eq_rc.py from tests/operation/test_eq_rc.py config.\n")
        f.write(f"`define GEN_TOL_NUM {num}\n")
        f.write(f"`define GEN_TOL_DEN {den}\n")
        f.write(f"`define GEN_TW {tw}\n")
    print(f"wrote {VEC} ({len(rows_u)} vectors); {HDR} (NUM={num} DEN={den} TW={tw})")


if __name__ == "__main__":
    main()
