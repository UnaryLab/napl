"""
Shared helper for the per-op golden-vector generators (`<op>/gen/gen_<op>.py`).

The core rule: the per-cycle spike streams driven into the RTL must be the
SAME streams the op's `test_<op>.py` produces, i.e. encoded by the test's encoder
config (polarity / timestep / generator / sobol dim), not an ad hoc random or
hand-built bit pattern. napl encoders are deterministic, so rebuilding an encoder
from the test's exact `codec_config` and feeding it a scalar value reproduces the
identical per-timestep spike stream the test sends the op.

`encode_value(codec_config, value)` returns the list of 0/1 spikes (length =
timestep) that `encode(codec_config)` emits for `value`, by driving the real
napl encoder over its timestep count from a fresh reset. `rep_values(polarity)`
gives the representative scalar operands to encode: the known-answer corners plus
a small in-range sweep plus a few deterministic distribution draws, matching the
regime the test's `gen_rand_tensor` covers.
"""
import torch

from napl.sim.operation import encode


def encode_value(codec_config, value):
    """Per-cycle spike list (0/1) the test's encoder emits for `value`.

    Drives a real napl encoder built from `codec_config` over `timestep` cycles
    from a fresh reset, so the stream is bit-identical to what test_<op>.py sends.
    """
    enc = encode(dict(codec_config))
    enc.reset()
    v = torch.tensor(float(value)).type(enc.num_seq.dtype)
    spikes = []
    for _ in range(codec_config["timestep"]):
        spikes.append(int(enc(v).item()))
    return spikes


def pair_streams(codec0, codec1, value_pairs):
    """Two parallel per-cycle spike streams for (v0, v1) operand pairs.

    Each operand is encoded with its own test encoder config (distinct sobol
    dims keep the two streams decorrelated exactly as the test does). Segments
    for successive pairs are concatenated into two equal-length streams.
    """
    s0, s1 = [], []
    for v0, v1 in value_pairs:
        s0 += encode_value(codec0, v0)
        s1 += encode_value(codec1, v1)
    return s0, s1


def rep_pairs(p0, p1, n=8, seed=0, range0=None, range1=None):
    """Representative (v0, v1) operand pairs to encode for two-input ops.

    Mixes corner combinations (rails of each operand) with deterministic
    in-range draws, covering equal / v0<v1 / v0>v1 regimes.
    """
    a = rep_values(p0, n_draw=3, seed=seed, value_range=range0)
    b = rep_values(p1, n_draw=3, seed=seed + 1, value_range=range1)
    pairs = []
    for x in a[:5]:
        for y in b[:5]:
            pairs.append((x, y))
    g = torch.Generator().manual_seed(seed + 7)
    lo0, hi0 = (range0 if range0 else ((-1, 1) if p0 == "bipolar" else (0, 1)))
    lo1, hi1 = (range1 if range1 else ((-1, 1) if p1 == "bipolar" else (0, 1)))
    for _ in range(n):
        x = lo0 + (hi0 - lo0) * torch.rand(1, generator=g).item()
        y = lo1 + (hi1 - lo1) * torch.rand(1, generator=g).item()
        pairs.append((x, y))
    return pairs


def rep_values(polarity, n_draw=4, seed=0, value_range=None):
    """Representative scalar operands to encode with the test's config.

    corners (known-answer rails + midpoint) + an in-range sweep + a few
    deterministic distribution draws. Range follows the encoder polarity by
    default: unipolar -> [0,1], bipolar -> [-1,0.5], whose probabilities
    [0,0.75] land off the unipolar grid, so a generator that builds both
    polarity streams from this helper gets distinct stimulus per polarity.
    Pass `value_range=(lo,hi)` to override when the test draws a different
    operand range than its encoder polarity (e.g. sqrt feeds unipolar-range
    [0,1] values to a bipolar encoder).
    """
    if value_range is None:
        value_range = (-1.0, 0.5) if polarity == "bipolar" else (0.0, 1.0)
    lo, hi = value_range
    mid = (lo + hi) / 2
    corners = [lo, (lo + mid) / 2, mid, (mid + hi) / 2, hi]
    sweep = [lo + (hi - lo) * k / 6 for k in range(1, 6)]
    g = torch.Generator().manual_seed(seed)
    draws = (lo + (hi - lo) * torch.rand(n_draw, generator=g)).tolist()
    out, seen = [], set()
    for x in corners + sweep + draws:
        key = round(x, 6)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out
