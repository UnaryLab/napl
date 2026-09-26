"""
Generate golden test vectors for the encode_hold RTL straight from napl's
functional Python model (napl.sim.operation.encode_hold), so the testbench checks
the Verilog against the actual simulator, not a hand-derived truth table.

encode_hold latches the decoded estimate of the first `trigger_timestep` input
spikes and thereafter re-emits a fresh Sobol-encoded stream of that frozen value.
The re-encode probability is count_k / TRIGGER for both polarities, so one bare
module covers both; this generator drives full-legal-range values per polarity
through it, mirroring test_encode_hold.py's timestep (256) and trigger (timestep
// 2 = 128).

The re-encoder's number sequence is generated online in hardware by the Sobol
generator inside the encode sub-instance. This file recovers that dimension's
direction-vector table from the model's own num_seq, asserts the gray-code
recurrence reproduces the whole 2**WIDTH-point sequence, and writes it to
../vec/encode_dirvec_d1.hex, the default DIRVEC_FILE the sub-instance reads
relative to the simulation cwd. The re-encode probability is carried in
`frac` = WIDTH + 1 fractional bits, where count_k / TRIGGER and every sequence
value are exactly representable and the integer compare reproduces the model's
float compare bit for bit.

Output: ../vec/encode_hold.vec, one line per timestep:

    <rst> <i_input> <o_output>   (rst=1 means "reset BEFORE this cycle"; the spike
                                  and output are 0/1)

The sizing params WIDTH and TRIGGER are the single source of truth: derived from
config once, used to build the model, and emitted into ../vec/encode_hold_params.vh
so the testbench overrides the RTL parameters with the same values.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_encode_hold.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import encode, encode_hold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "encode_hold.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "encode_hold_params.vh"
DIRVEC = Path(__file__).resolve().parent.parent / "vec" / "encode_dirvec_d1.hex"

# Sizing mirrors test_encode_hold.py: timestep 256, trigger = timestep // 2 = 128.
TIMESTEP = 256
TRIGGER = TIMESTEP // 2
GENERATOR = "sobol"


def dirvec_rows(num_seq, width, name):
    """Direction vectors of a Sobol sequence, checked against the model's num_seq.

    The hardware generator runs the Antonov-Saleev gray-code recurrence
    x_{n+1} = x_n ^ v[l(n)], where l(n) is the position of the least significant
    zero of the WIDTH-bit counter n. The table v is recovered from the model's own
    sequence and replayed here over the whole period, so a sequence the recurrence
    does not reproduce fails the generator instead of the co-simulation.
    """
    period = 2 ** width
    values = num_seq.detach().float().reshape(-1)
    assert values.numel() == period, f"{name} holds {values.numel()} points, not {period}"
    codes = []
    for index in range(period):
        scaled = values[index].item() * period
        code = round(scaled)
        assert abs(scaled - code) < 1e-9, f"{name}[{index}] is off the 1/{period} grid"
        codes.append(code)
    assert codes[0] == 0, f"{name} starts at {codes[0]}, not the post-reset 0"

    vectors = [codes[2 ** k] ^ codes[2 ** k - 1] for k in range(width)]
    state = 0
    for index in range(period):
        assert state == codes[index], \
            f"{name} is not a gray-code Sobol sequence: the recurrence gives {state} " \
            f"at index {index}, the model gives {codes[index]}"
        # The all-ones counter state takes the top position, which returns to 0.
        position = width - 1 if index == period - 1 else (~index & (index + 1)).bit_length() - 1
        state ^= vectors[position]
    assert state == 0, f"{name} returns to {state} on the wrap, not 0"
    return [f"{vector:0{width}b}" for vector in vectors]


def write_dirvec(model, width):
    """Emit the re-encoder's Sobol direction vectors."""
    rows = dirvec_rows(model.reference_encode.num_seq, width, "re-encoder sequence")
    DIRVEC.write_text("\n".join(rows) + "\n")
    return len(rows)


def main():
    assert TRIGGER & (TRIGGER - 1) == 0, (
        f"trigger_timestep {TRIGGER} must be a power of two for a bit-exact circuit")

    VEC.parent.mkdir(parents=True, exist_ok=True)

    ref = encode_hold({"polarity": "bipolar", "timestep": TIMESTEP,
                       "trigger_timestep": TRIGGER, "generator": GENERATOR})
    width = ref.decoder.width
    dirvec_lines = write_dirvec(ref, width)

    rows = []
    for polarity in ("bipolar", "unipolar"):
        op = encode_hold({"polarity": polarity, "timestep": TIMESTEP,
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
    PARAMS.write_text(
        f"`define GEN_WIDTH {width}\n"
        f"`define GEN_TRIGGER {TRIGGER}\n"
        f"`define GEN_PP_DELAY {ref.hw.pp_delay}\n"
        f"`define GEN_VECTORS {len(rows)}\n"
    )
    print(f"wrote {VEC} ({len(rows)} vectors), {PARAMS} "
          f"(GEN_WIDTH={width}, GEN_TRIGGER={TRIGGER}), and {DIRVEC} "
          f"({dirvec_lines} direction vectors)")


if __name__ == "__main__":
    main()
