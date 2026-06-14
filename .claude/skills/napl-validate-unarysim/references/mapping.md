# napl to UnarySim mapping

UnarySim is `github.com/diwu1990/UnarySim`; the canonical reference is the local clone at
`/Users/diwu/Projects/UnarySim`. Every class name and file below was **verified against that clone**
(re-verified 2026-06-13 by listing `^class` in each file). When in doubt, re-list the source rather
than trusting this table, since UnarySim reorganizes over time:

```bash
grep -rnoE '^class +[A-Za-z0-9_]+' kernel stream metric   # run inside the clone
```

A napl idea (encode -> op -> decode over timesteps, then `report_error`) corresponds to the UnarySim
`RNG`/`RawScale`/`BinGen`/`BSGen` -> kernel -> `ProgError` pipeline.

## UnarySim repo layout (verified class lists)

- `kernel/` - one file per op family:
  - `add.py` (`FSUAdd`), `mul.py` (`FSUMul`), `div.py` (`CORDIV_kernel`, `FSUDiv`),
    `sqrt.py` (`FSUSqrt`), `signabs.py` (`FSUSignAbs`)
  - `relu.py` (`FSUReLU`, `HUBReLU`), `sigmoid.py` (`FSUHardsigmoid`, `HUBHardsigmoid`),
    `tanh.py` (`FSUHardtanh`, `HUBHardtanh`)
  - `jkff.py` (`JKFF`), `shiftreg.py` (`ShiftReg`)
  - `linear.py` (`FSULinear`, `FSULinearPC`, `HUBLinear`, `FXPLinear`, `TLUTLinear` + their
    autograd `*Function`s), `conv.py` (`FSUConv2d`, `FSUConv2dPC`, `HUBConv2d`, `FXPConv2d`,
    `TLUTConv2d`), `rnn.py` (`FSUMGUCell`, `HUBMGUCell`, `HardMGUCell`, `HardMGUCellFXP`)
  - `utils.py` (`NN_SC_Weight_Clipper`, `RoundSTE`, `Round`)
- `metric/metric.py` - `Correlation`, `ProgError`, `Stability`
- `stream/gen.py` - RNG / bitstream generation: `RNG`, `RawScale`, `BinGen`, `BSGen`
- `stream/shuffle.py` - stream reshapers: `SkewedSync`, `Bi2Uni`, `Uni2Bi`
  (the polarity converters and the skewed synchronizer live **here, not in `kernel/`**)

**UnarySim classes not (yet) ported to napl:** *(none of the listed kernels remain unported.)*
(`FSULinearPC` is ported as `linear_fsu_pc` and `FSUConv2dPC` as `conv_fsu_pc`; see the
neural-layers table.)

## Metrics (`napl.metric`  ->  UnarySim `metric/metric.py`)

| napl | UnarySim class | report method (napl / UnarySim) |
|------|----------------|---------------------------------|
| `correlation` | `Correlation` | `report_corr()` / `forward()` (update via `Monitor()`) |
| `stability`   | `Stability`   | `report_stab()` / `forward()` (update via `Monitor()`) |
| `accuracy`    | `ProgError`   | `report_error()` / `forward()` (update via `Monitor()`) |

## Operations / gates (`napl.operation`  ->  UnarySim)

| napl class | UnarySim class | file | notes |
|------------|----------------|------|-------|
| `mul_csg` | `FSUMul(static=True)` | `kernel/mul.py` | exact module match; carries an RNG -> SC-bound, see note |
| `mul_and` | gate core inside `FSUMul` (no standalone module) | `kernel/mul.py` | bit-exact gate identity, see note |
| `add_any` | `FSUAdd` | `kernel/add.py` | |
| `div_cordiv` | `CORDIV_kernel` | `kernel/div.py` | correlated division, unipolar; operands pre-synchronized |
| `div_iscb` | `FSUDiv` | `kernel/div.py` | in-stream correlation-based division (iscbdiv) |
| `sqrt_tracejkff` | `FSUSqrt(jk_trace=True, emit=False)` | `kernel/sqrt.py` | one UnarySim class, flag-selected; napl split into 3 modules |
| `sqrt_traceiscb` | `FSUSqrt(jk_trace=False, emit=False)` | `kernel/sqrt.py` | iscbdiv-based trace |
| `sqrt_emit` | `FSUSqrt(emit=True)` | `kernel/sqrt.py` | opportunistic bit-inserting |
| `sign_abs` | `FSUSignAbs` | `kernel/signabs.py` | |
| `relu_cnt`, `relu_sat` | `FSUReLU` | `kernel/relu.py` | two napl variants of one UnarySim FSU module |
| `relu_hub` | `HUBReLU` | `kernel/relu.py` | binary-domain |
| `sigmoid_hard` / `sigmoid_hub` | `FSUHardsigmoid` / `HUBHardsigmoid` | `kernel/sigmoid.py` | |
| `tanh_hard` / `tanh_hub` | `FSUHardtanh` / `HUBHardtanh` | `kernel/tanh.py` | |
| `jkff` | `JKFF` | `kernel/jkff.py` | |
| `shiftreg` | `ShiftReg` | `kernel/shiftreg.py` | |
| `sync_skewed` | `SkewedSync` | `stream/shuffle.py` | **not** in `kernel/` |
| `bi2uni` / `uni2bi` | `Bi2Uni` / `Uni2Bi` | `stream/shuffle.py` | **not** in `kernel/` |
| `round_ste` / `round_fxp` | `RoundSTE` / `Round` | `kernel/utils.py` | `round.py` also has `_round_ste_fn` (autograd) |
| `dff` | *(no standalone module)* | - | UnarySim has no DFF kernel; closest is a depth-1 `ShiftReg` delay. Validate as the delay identity |
| `square_dff` | *(no standalone module)* | - | UnarySim has **no** `FSUSquare`; napl builds square from AND + `dff` (uGEMM). Validate against AND-of-decorrelated-stream math, not an UnarySim class |
| `min_rc`, `max_rc`, `lt_rc`, `gt_rc` | *(no standalone module)* | - | rate-coded compare, built on `sync_skewed`/`SkewedSync`; UnarySim does not expose these as classes |
| `min_tc`, `max_tc` | *(no standalone module)* | - | temporal-coded min/max (elementwise on temporal streams); no UnarySim class |
| *(napl `inhibit`)* | *(placeholder)* | - | `operation/inhibit.py` is empty; nothing to map yet |

### Note: `mul_and` vs `mul_csg` vs `FSUMul` (verified 2026-06-13)

`FSUMul` is not a bare gate. Its `forward` always combines the streamed operand `in_0` with a
**second operand it generates internally**, in one of two modes (`hwcfg["static"]`):

- `static=True`: `in_1` is a fixed value `in_1_prob`; FSUMul makes its spike with `BSGen`+`RNG` and a
  conditional `rng_idx` enable update, then `in_0 & bsg(...)` (bipolar adds the inverse path). This is
  **exactly napl `mul_csg`** (streamed operand x value operand, conditional spike generation). It
  carries an RNG, so napl-vs-UnarySim is SC-bound, not bit-exact, unless the two RNGs coincide.
- `static=False` (in-stream): `in_1` is streamed but decorrelated through a `ShiftReg`, then
  `in_0 & gt(source, rng[idx])`. This in-stream multiplier has **no direct napl equivalent** in the
  current op set.

napl `mul_and` is the **pure AND (unipolar) / XNOR (bipolar) gate of two already-decorrelated spike
streams** - no RNG. That gate is the `&`/`xnor` core *inside* `FSUMul_forward`, but UnarySim does not
expose it as a standalone module (FSUMul always wraps it with internal spike generation). So validate
`mul_and` as the gate identity (`in_0 & in_1`; bipolar `1 - (in_0 ^ in_1)` == napl's
`xor(a,b).xor_(1)`) on two externally encoded, distinct-Sobol-dim streams - it is bit-exact by
construction. Validate `mul_csg` against `FSUMul(static=True)`.

### Note: napl ops with no UnarySim counterpart

Several napl ops are napl-native (composed from primitives, or added after the port) and have **no
standalone UnarySim class** to diff against. For these, validate against the *mathematical* identity
(or the napl primitive they wrap), not an UnarySim module:

- `dff` - a unit delay; UnarySim only has `ShiftReg` (a depth-N register) and `JKFF`. Check that
  `dff(depth=1)` reproduces a one-timestep shift.
- `square_dff` - unary square = `in & delay(in)` (uGEMM), built from `dff`. UnarySim has no
  `FSUSquare` module; validate against the AND-of-decorrelated-stream analytic result.
- `compare` family (`min_rc`/`max_rc`/`lt_rc`/`gt_rc`/`min_tc`/`max_tc`) - rate-coded versions use
  `sync_skewed` (= `SkewedSync`) then a gate; temporal versions are elementwise. No UnarySim classes.
- `inhibit` - placeholder (empty file).

## Neural layers (`napl.module`  ->  UnarySim `kernel/{linear,conv,rnn}.py`)

| napl | UnarySim class | file | notes |
|------|----------------|------|-------|
| `linear_fsu` | `FSULinear` | `kernel/linear.py` | |
| `linear_fsu_pc` | `FSULinearPC` | `kernel/linear.py` | parallel-counter (per-step PC count, no accumulator); independent decorrelated encoders vs FSULinearPC's CSG weight indexing, agrees within SC bound |
| `linear_hub` / `linear_fxp` / `linear_tlut` | `HUBLinear` / `FXPLinear` / `TLUTLinear` | `kernel/linear.py` | |
| `conv_fsu` | `FSUConv2d` | `kernel/conv.py` | |
| `conv_fsu_pc` | `FSUConv2dPC` | `kernel/conv.py` | parallel-counter (per-step PC count, no accumulator); independent decorrelated weight/bias encoders, groups=1 zero-padding only, agrees within SC bound |
| `conv_hub` / `conv_fxp` / `conv_tlut` | `HUBConv2d` / `FXPConv2d` / `TLUTConv2d` | `kernel/conv.py` | |
| `mgu_fsu` | `FSUMGUCell` | `kernel/rnn.py` | |
| `mgu_hub` / `mgu_hard` / `mgu_hardfxp` | `HUBMGUCell` / `HardMGUCell` / `HardMGUCellFXP` | `kernel/rnn.py` | |

napl's `module/wta.py` is a placeholder (no class yet); UnarySim has no WTA module either.

## utils helpers (`napl.utils`  ->  UnarySim `kernel/utils.py`)

| napl | UnarySim |
|------|----------|
| `rshift_offset` | `rshift_offset` |
| `conv2d_output_shape`, `conv2d_get_padding` | same names |
| `truncated_normal` | `truncated_normal` |
| `NN_SC_Weight_Clipper` | `NN_SC_Weight_Clipper` |
| `pow2_lshift` / `pow2_rshift` | the RAVEN float-shift behavior these shims replace |

## codec (`napl.module`  ->  UnarySim `stream/gen.py`)

`encoder` / `decoder` / `gen_num_seq` correspond to UnarySim's `RNG` + `RawScale` + `BinGen` +
`BSGen` bitstream pipeline (note: UnarySim's old `SourceGen` is now `BinGen`). There is no single
1:1 class; validate the round-trip (encode -> decode of a known value) rather than a single call.
