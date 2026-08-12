"""
Generate golden test vectors for the pow_n RTL straight from napl's functional
Python model (napl.sim.operation.pow_n) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

pow_n raises one spike stream to the integer power N: it multiplies the stream
against N-1 decorrelated (delayed) copies of itself, one Gaines multiply per
copy, with stage k using a depth-k dff (delays 1..N-1). The dff chain is the
only state and is reset to all-zeros, so we drive a multi-cycle stream from a
fresh reset and record the per-cycle (input, output) for both polarity variants.
A single spike stream feeds both models: the RTL is bit-level, so bit-exactness
only needs the same input bits, and both polarity DUTs share i_input in the tb.
A MID-STREAM reset (rst=1) is injected to prove reset equivalence from a dirtied
state: model.reset() is replayed there and the RTL's active-low i_rst_n pulsed,
and both must resume from the cleared dff chain identically.

The sizing param N is the single source of truth here: it is read ONCE from the
op config (mirroring test_pow_n.py's N), used to build the model, AND emitted
into ../vec/pow_n_params.vh as `GEN_N so the testbench overrides the RTL
parameter with the same value. RTL and sim therefore inherit N from one place;
they cannot drift.

Output: ../vec/pow_n.vec, one line per timestep:

    <rst> <i_input> <out_unipolar> <out_bipolar>
    (each 0/1, space-separated)

rst=1 marks cycles where reset() is applied (to the model) / i_rst_n pulsed low
(in the RTL) BEFORE driving i_input.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_pow_n.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import pow_n
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "pow_n.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "pow_n_params.vh"

# Power exercised, mirroring test_pow_n.py's N. x^N stays in the legal range for
# both polarities: [0, 1] -> [0, 1] unipolar, [-1, 1] -> [-1, 1] bipolar.
N = 3
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}

# One shared input stream at the codec's fidelity scale, over the full legal
# range; x^N stays in range for both polarities. Split to reset mid-stream.
_VALUES = rep_values(CODEC["polarity"])
_MID = len(_VALUES) // 2

DRIVE = []
for _i, _v in enumerate(_VALUES):
    _seg = encode_value(CODEC, _v)
    for _j, _s in enumerate(_seg):
        _rst = 1 if (_i == _MID and _j == 0) else 0
        DRIVE.append((_rst, _s))


def run(polarity):
    model = pow_n(config={"polarity": polarity, "n": N})
    model.reset()
    outs = []
    for rst, s in DRIVE:
        if rst:
            model.reset()
        spike = torch.tensor(s, dtype=model.stype)
        outs.append(int(model(spike).item()))
    return outs


def main():
    out_uni = run("unipolar")
    out_bi = run("bipolar")

    # Resolve each elaborated variant's mapping entry and require it to reproduce
    # the param these vectors were built with, so make test gates the mapping too.
    for polarity in ("unipolar", "bipolar"):
        binding = translate_node({"class": "pow_n", "config": {"polarity": polarity, "n": N}})
        assert binding.parameters == {"N": N}, \
            f"mapping pow_n_{polarity} resolves {binding.parameters}, not {{'N': {N}}}"

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_N {N}\n")

    with VEC.open("w") as f:
        for (rst, s), u, b in zip(DRIVE, out_uni, out_bi):
            f.write(f"{rst} {s} {u} {b}\n")
    print(f"wrote {VEC} ({len(DRIVE)} vectors) and {PARAMS} (GEN_N={N})")


if __name__ == "__main__":
    main()
