"""
Generate golden test vectors for the sample_hold RTL straight from napl's
functional Python model (napl.sim.operation.sample_hold), so the testbench checks
the Verilog against the actual simulator, not a hand-derived truth table.

sample_hold latches the decoded estimate of the first `trigger_timestep` input
spikes and thereafter re-emits a fresh Sobol-encoded stream of that frozen value.
The re-encode probability is count_k / TRIGGER for both polarities, so one bare
module covers both; this generator drives full-legal-range values per polarity
through it, mirroring test_sample_hold.py's timestep (256) and trigger (timestep
// 2 = 128).

The re-encoder's number sequence is data, not logic: it is emitted here from the
model into ../vec/encode_rom.hex (the default ROM_FILE the encode sub-instance
reads, relative to the simulation cwd) with `frac` = WIDTH + 1 fractional bits,
where count_k / TRIGGER and every sequence value are exactly representable and the
integer compare reproduces the model's float compare bit for bit.

Output: ../vec/sample_hold.vec, one line per timestep:

    <rst> <i_input> <o_output>   (rst=1 means "reset BEFORE this cycle"; the spike
                                  and output are 0/1)

The sizing params WIDTH and TRIGGER are the single source of truth: derived from
config once, used to build the model, and emitted into ../vec/sample_hold_params.vh
so the testbench overrides the RTL parameters with the same values.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_sample_hold.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import encode, sample_hold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "sample_hold.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "sample_hold_params.vh"
ROM = Path(__file__).resolve().parent.parent / "vec" / "encode_rom.hex"

# Sizing mirrors test_sample_hold.py: timestep 256, trigger = timestep // 2 = 128.
TIMESTEP = 256
TRIGGER = TIMESTEP // 2
GENERATOR = "sobol"


def write_rom(model, frac):
    """Emit the re-encoder's number sequence as frac-bit binary ROM lines."""
    seq = model.reference_encode.num_seq.detach().float().reshape(-1)
    lines = []
    for i in range(seq.numel()):
        scaled = seq[i].item() * 2 ** frac
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f"num_seq[{i}]={seq[i].item()} off the 1/2**{frac} grid"
        assert 0 <= code < 2 ** frac, f"sequence code {code} out of [0, 2**{frac})"
        lines.append(f"{code:0{frac}b}")
    ROM.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    assert TRIGGER & (TRIGGER - 1) == 0, (
        f"trigger_timestep {TRIGGER} must be a power of two for a bit-exact circuit")

    VEC.parent.mkdir(parents=True, exist_ok=True)

    ref = sample_hold({"polarity": "bipolar", "timestep": TIMESTEP,
                       "trigger_timestep": TRIGGER, "generator": GENERATOR})
    width = ref.decoder.width
    frac = width + 1
    rom_lines = write_rom(ref, frac)
    PARAMS.write_text(
        f"`define GEN_WIDTH {width}\n"
        f"`define GEN_TRIGGER {TRIGGER}\n"
        f"`define GEN_PP_DELAY {ref.hw.pp_delay}\n"
    )

    rows = []
    for polarity in ("bipolar", "unipolar"):
        op = sample_hold({"polarity": polarity, "timestep": TIMESTEP,
                          "trigger_timestep": TRIGGER, "generator": GENERATOR})
        # Full legal range per polarity, encoded exactly as the streaming suite feeds
        # the op: sobol dim-1 rate coding at the same timestep.
        codec = {"polarity": polarity, "timestep": TIMESTEP,
                 "generator": GENERATOR, "dim": 1}
        for value in rep_values(polarity):
            enc_in = encode(codec)
            op.reset()
            enc_in.reset()
            x = torch.tensor([float(value)]).type(enc_in.num_seq.dtype)
            for t in range(TIMESTEP):
                rst = 1 if t == 0 else 0
                spike = enc_in(x)
                out = int(op(spike).item())
                rows.append(f"{rst} {int(spike.item())} {out}")

    VEC.write_text("\n".join(rows) + "\n")
    print(f"wrote {VEC} ({len(rows)} vectors), {PARAMS} "
          f"(GEN_WIDTH={width}, GEN_TRIGGER={TRIGGER}), and {ROM} ({rom_lines} ROM lines)")


if __name__ == "__main__":
    main()
