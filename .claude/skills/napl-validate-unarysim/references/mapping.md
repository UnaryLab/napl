# napl to UnarySim mapping

UnarySim is `github.com/diwu1990/UnarySim`; the canonical reference is the local clone at
`/Users/diwu/Projects/UnarySim` at commit `0304237`. Every class name and file below was
**verified against that clone** on 2026-07-25 by listing `^class` in each file. When in doubt,
re-list the source rather than trusting this table, since UnarySim reorganizes over time:

```bash
rg -n '^class +[A-Za-z0-9_]+' kernel stream metric   # run inside the clone
```

A napl idea (encode -> op -> accuracy observer over timesteps, then `analyze(reference)`) corresponds to the UnarySim
`RNG`/`RawScale`/`SourceGen`/`BSGen` -> kernel -> `ProgError` pipeline.

## UnarySim repo layout (verified class lists)

- `kernel/` - one file per op family:
  - `add.py` (`FSUAdd`), `mul.py` (`FSUMul`), `div.py` (`CORDIV_kernel`, `FSUDiv`),
    `sqrt.py` (`FSUSqrt`), `abs.py` (`FSUAbs`), `sign.py` (`FSUSign`)
  - `relu.py` (`FSUReLU`, `ScaleReLU`),
    `sigmoid.py` (`FSUHardsigmoid`, `ScaleHardsigmoid`),
    `tanh.py` (`FSUHardtanh`, `ScaleHardtanh`)
  - `comp.py` (`FSUCompare`; unusable dead code, see the operations table)
  - `jkff.py` (`JKFF`), `shiftreg.py` (`ShiftReg`)
  - `linear.py` (`FSULinear`, `FSULinearPC`, `HUBLinear`, `FxpLinear` + their autograd
    `*Function`s), `conv.py` (`FSUConv2d`, `FSUConv2dPC`, `HUBConv2d`, `FxpConv2d`),
    `rnn.py` (`FSUMGUCell`, `HUBMGUCell`, `HardMGUCell`, `HardMGUCellFxp`)
  - `utils.py` (`NN_SC_Weight_Clipper`, `RoundingNoGrad`, `Round`)
- `metric/metric.py` - `Correlation`, `ProgError`, `Stability`
- `stream/gen.py` - RNG / bitstream generation: `RNG`, `RNGMulti`, `RawScale`, `SourceGen`,
  `BSGen`, `BSGenMulti`
- `stream/shuffle.py` - stream reshapers: `SkewedSync`, `Bi2Uni`, `Uni2Bi`
  (the polarity converters and the skewed synchronizer live **here, not in `kernel/`**)

**UnarySim classes and modes not ported to napl:**

- `RNGMulti` and `BSGenMulti`
- `Decorr`, `Desync`, `Sync` (`stream/shuffle.py`) - unimplemented stubs upstream; their constructors `raise ValueError`, so there is nothing to port

## Metrics (`napl.sim.metric`  ->  UnarySim `metric/metric.py`)

| napl | UnarySim class | report method (napl / UnarySim) |
|------|----------------|---------------------------------|
| `correlation` | `Correlation` | `analyze()` / `forward()` (update via `Monitor()`) |
| `stability`   | `Stability`   | `analyze()` / `forward()` (update via `Monitor()`) |
| `accuracy`    | `ProgError`   | `analyze(reference)` / `forward()` (update via `Monitor()`) |
| `stability_norm`    | `NormStability` | `analyze()` / `forward()` (update via `Monitor()`) |
| `stability_builder` | `NSbuilder`     | build-then-emit; see class docs |
| `stability_flux`    | *(no upstream class)* | `analyze()`; napl-native ratio of two `Stability` monitors; validate against `S1.forward() / S2.forward()` fed identical streams |

## Operations / gates (`napl.sim.operation`  ->  UnarySim)

| napl class | UnarySim class | file | notes |
|------------|----------------|------|-------|
| `mul_ugemm` | `FSUMul(static=True)` | `kernel/mul.py` | exact module match; carries an RNG -> SC-bound, see note |
| `mul_ugemm_sr` | `FSUMul(static=False)` | `kernel/mul.py` | in-stream shift-register decorrelation; bit-exact on identical input streams |
| `add_any` | `FSUAdd` | `kernel/add.py` | Intentional carry-threshold divergence: NAPL emits when accumulator >= scale; upstream FSUAdd emits only when accumulator > scale. |
| `div_cordiv` | `CORDIV_kernel` | `kernel/div.py` | correlated division, unipolar; operands pre-synchronized |
| `div_iscb` | `FSUDiv` | `kernel/div.py` | in-stream correlation-based division (iscbdiv) |
| `sqrt_tracejkff` | `FSUSqrt(jk_trace=True, emit=False)` | `kernel/sqrt.py` | one UnarySim class, flag-selected; napl split into 3 modules |
| `sqrt_traceiscb` | `FSUSqrt(jk_trace=False, emit=False)` | `kernel/sqrt.py` | iscbdiv-based trace |
| `sqrt_emit` | `FSUSqrt(emit=True)` | `kernel/sqrt.py` | opportunistic bit-inserting |
| `signabs` | `FSUAbs(shiftreg=False, interleave=False)` + `FSUSign(shiftreg=False)` | `kernel/abs.py`, `kernel/sign.py` | counter path; UnarySim exposes the absolute-value and sign functions as separate classes |
| `signabs_interleave` | `FSUAbs(interleave=True)` | `kernel/abs.py` | interleaved accumulator-parity magnitude path |
| `signabs_shiftreg` | `FSUAbs(shiftreg=True)` + `FSUSign(shiftreg=True)` | `kernel/abs.py`, `kernel/sign.py` | shift-register sign and magnitude path |
| `relu_cnt` | `FSUReLU(encode="RC", shiftreg=False)` | `kernel/relu.py` | counter path |
| `relu_shiftreg` | `FSUReLU(encode="RC", shiftreg=True)` | `kernel/relu.py` | shift-register path |
| `relu_tc` | `FSUReLU(encode="TC")` | `kernel/relu.py` | temporal-coded path; same decoded value and one-count, not bit-exact: napl emits a monotone temporal codeword, UnarySim emits an unconditional all-ones first half |
| `relu_sat` | `FSUReLU(encode="RC", shiftreg=False)` | `kernel/relu.py` | napl saturating-adder alternative; within SC bound, not bit-exact |
| `relu_hub` | `ScaleReLU` | `kernel/relu.py` | binary-domain |
| `sigmoid_hard` / `sigmoid_hub` | `FSUHardsigmoid` / `ScaleHardsigmoid` | `kernel/sigmoid.py` | |
| `tanh_hard` / `tanh_hub` | `FSUHardtanh` / `ScaleHardtanh` | `kernel/tanh.py` | |
| `jkff` | `JKFF` | `kernel/jkff.py` | |
| `shiftreg` | `ShiftReg` | `kernel/shiftreg.py` | |
| `sync_skewed` | `SkewedSync` | `stream/shuffle.py` | **not** in `kernel/` |
| `bi2uni` / `uni2bi` | `Bi2Uni` / `Uni2Bi` | `stream/shuffle.py` | **not** in `kernel/`; finite-width accumulator divergence: NAPL uses bounded accumulators and requires the emission threshold to be reachable, while the upstream accumulators are unbounded, so sustained stream imbalance can produce different per-timestep emission timing |
| `round_fxp` | `Round` | `kernel/utils.py` | the only public napl class; the upstream autograd function `RoundingNoGrad` is ported as the private `_round_ste_fn` inside `round_fxp.py` |
| `add_gaines` | `GainesAdd` | `kernel/add.py` | |
| `add_ugemm` | `FSUAdduGEMM` | `kernel/add.py` | |
| `mul_gaines` | `GainesMul` | `kernel/mul.py` | bit-exact gate identity, see note |
| `div_gaines` | `GainesDiv` | `kernel/div.py` | |
| `sqrt_gaines` | `GainesSqrt` | `kernel/sqrt.py` | |
| `tanh_p1` / `tanh_pn` | `tanhP1` / `tanhPN` | `kernel/tanh.py` | |
| `exp_n1` / `exp_n2g` | `expN1` / `expNG` | `kernel/exp.py` | |
| `sync_skewed_int` | `SkewedSyncInt` | `stream/shuffle_int.py` | **not** in `kernel/` |
| `dff` | *(no standalone module)* | - | UnarySim has no DFF kernel; closest is a depth-1 `ShiftReg` delay. Validate as the delay identity |
| `square_dff` | *(no standalone module)* | - | UnarySim has **no** `FSUSquare`; napl builds square from AND + `dff` (uGEMM). Validate against AND-of-decorrelated-stream math, not an UnarySim class |
| `min_rc`, `max_rc`, `lt_rc`, `gt_rc` | *(no usable upstream counterpart)* | - | `FSUCompare` is dead code: its forward path ignores `in_1` and `in_2` and references the builtin `input` |
| `min_tc`, `max_tc` | *(no standalone module)* | - | temporal-coded min/max (elementwise on temporal streams); no UnarySim class |
| `inhibit` | *(no standalone module)* | - | race-logic INHIBIT gate (Boosted Race Trees, ASPLOS'19 Fig 3); no UnarySim class |

### Note: `mul_gaines` vs `mul_ugemm` vs `FSUMul` (verified 2026-06-13)

`FSUMul` is not a bare gate. Its `forward` always combines the streamed operand `in_0` with a
**second operand it generates internally**, in one of two modes (`hwcfg["static"]`):

- `static=True`: `in_1` is a fixed value `in_1_prob`; FSUMul makes its spike with `BSGen`+`RNG` and a
  conditional `rng_idx` enable update, then `in_0 & bsg(...)` (bipolar adds the inverse path). This is
  **exactly napl `mul_ugemm`** (streamed operand x value operand, conditional spike generation). It
  carries an RNG, so napl-vs-UnarySim is SC-bound, not bit-exact, unless the two RNGs coincide.
- `static=False` (in-stream): `in_1` is streamed but decorrelated through a `ShiftReg`, then
  `in_0 & gt(source, rng[idx])`. This is napl `mul_ugemm_sr`.

napl `mul_gaines` is the **pure AND (unipolar) / XNOR (bipolar) gate of two already-decorrelated
spike streams** - no RNG. UnarySim exposes that gate as the standalone `GainesMul`, and it is also the
`&`/`xnor` core *inside* `FSUMul_forward`, which FSUMul always wraps with internal spike generation.
So validate `mul_gaines` as the gate identity (`in_0 & in_1`; bipolar `1 - (in_0 ^ in_1)` == napl's
`xor(a,b).xor_(1)`) on two externally encoded, distinct-Sobol-dim streams - it is bit-exact by
construction. Validate `mul_ugemm` against `FSUMul(static=True)`.

### Note: napl ops with no UnarySim counterpart

Several napl ops are napl-native (composed from primitives, or added after the port) and have **no
standalone UnarySim class** to diff against. For these, validate against the *mathematical* identity
(or the napl primitive they wrap), not an UnarySim module:

- `dff` - a unit delay; UnarySim only has `ShiftReg` (a depth-N register) and `JKFF`. Check that
  `dff(depth=1)` reproduces a one-timestep shift.
- `square_dff` - unary square = `in & delay(in)` (uGEMM), built from `dff`. UnarySim has no
  `FSUSquare` module; validate against the AND-of-decorrelated-stream analytic result.
- `compare` family (`min_rc`/`max_rc`/`lt_rc`/`gt_rc`/`min_tc`/`max_tc`) - rate-coded versions use
  `sync_skewed` (= `SkewedSync`) then a gate; temporal versions are elementwise. UnarySim's
  `FSUCompare` is unusable dead code because it ignores both forward inputs and references the builtin `input`.
- `inhibit` - race-logic INHIBIT gate from *Boosted Race Trees for Low Energy Classification*
  (ASPLOS 2019, Fig 3): the data stream passes when its rising edge arrives no later than the
  inhibiting stream's; a strictly earlier inhibitor latches the output to never fire (all zeros,
  the minimum value under napl's larger-value-fires-earlier temporal code). Validate against
  `where(x0 >= x1, x0, min)` on temporal streams, not an UnarySim class.

## Neural layers (`napl.sim.module`  ->  UnarySim `kernel/{linear,conv,rnn}.py`)

| napl | UnarySim class | file | notes |
|------|----------------|------|-------|
| `linear` | `FSULinear` | `kernel/linear.py` | |
| `linear_pc` | `FSULinearPC` | `kernel/linear.py` | parallel-counter (per-step PC count, no accumulator); independent decorrelated encoders vs FSULinearPC's CSG weight indexing, agrees within SC bound |
| `linear_hub` | `HUBLinear` | `kernel/linear.py` | |
| `linear_fxp` | `FxpLinear` | `kernel/linear.py` | Intentional multi-call mapping divergence: NAPL recomputes input, weight, and output quantization shifts on every forward call, while UnarySim caches them after the first call; quantized results can differ when input ranges change |
| `linear_tlut` | *(no upstream counterpart)* | - | UnarySim has no TLUT linear class |
| `conv` | `FSUConv2d` | `kernel/conv.py` | |
| `conv_pc` | `FSUConv2dPC` | `kernel/conv.py` | parallel-counter (per-step PC count, no accumulator); independent decorrelated weight/bias encoders, groups=1 zero-padding only, agrees within SC bound |
| `conv_hub` | `HUBConv2d` | `kernel/conv.py` | |
| `conv_fxp` | `FxpConv2d` | `kernel/conv.py` | Intentional multi-call mapping divergence: NAPL recomputes input, weight, and output quantization shifts on every forward call, while UnarySim caches them after the first call; quantized results can differ when input ranges change |
| `conv_tlut` | *(no upstream counterpart)* | - | UnarySim has no TLUT convolution class |
| `mgu` | `FSUMGUCell` | `kernel/rnn.py` | |
| `mgu_hub` / `mgu_hard` / `mgu_hardfxp` | `HUBMGUCell` / `HardMGUCell` / `HardMGUCellFxp` | `kernel/rnn.py` | |
| `linear_ugemm` | `FSULinearuGEMM` | `kernel/linear.py` | |
| `linear_gaines1` / `linear_gaines2` | `GainesLinear1` / `GainesLinear4` | `kernel/linear.py` | `linear_gaines2` ports `GainesLinear4`; `GainesLinear2` and `GainesLinear3` have no napl counterpart |
| `conv_ugemm` | `FSUConv2duGEMM` | `kernel/conv.py` | |
| `avgpool2d` | `FSUAvgPool2d` | `kernel/pool.py` | |
| `mgu_hardnua` / `mgu_hardpt` | `HardMGUCellNUA` / `HardMGUCellPT` | `kernel/rnn.py` | |
| `gru_hardnuapt` | `HardGRUCellNUAPT` | `kernel/rnn.py` | |

napl's `module/wta.py` is a placeholder (no class yet); UnarySim has no WTA module either.

## utils helpers (`napl.utils`  ->  UnarySim `kernel/utils.py`)

| napl | UnarySim |
|------|----------|
| `rshift_offset` | *(no standalone function)* - the logic is inlined in `HUBLinear.forward` / `FxpLinear.forward`; validate against that inlined computation |
| `conv2d_output_shape`, `conv2d_get_padding` | same names |
| `truncated_normal` | `truncated_normal` |
| `NN_SC_Weight_Clipper` | `NN_SC_Weight_Clipper` |
| `pow2_lshift` / `pow2_rshift` | the RAVEN float-shift behavior these shims replace |

## codec (`napl.sim.operation`  ->  UnarySim `stream/gen.py`)

`encode` / `decode` / `gen_num_seq` correspond to UnarySim's `RNG` + `RawScale` + `SourceGen` +
`BSGen` bitstream pipeline. There is no single 1:1 class; validate the round-trip
(encode -> decode of a known value) rather than a single call.
