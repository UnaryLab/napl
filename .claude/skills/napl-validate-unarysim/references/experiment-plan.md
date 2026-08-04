# Experiment plan: port all UnarySim metrics & kernels into napl

**Status:** approved 2026-06-12. Phase B (build) in progress.

## Goal

Expand napl to support all metrics and kernels from UnarySim main
(github.com/diwu1990/UnarySim) that are not already present. This is a faithful
**port** of an existing functional simulator's math into napl's conventions, not a
new-claim experiment.

## Context note

Project memory (`napl-unarysim-kernel-port`, 2026-05-27) claimed this port already
existed (47/47 green). It exists nowhere in git (no branch/stash/commit/reflog);
`module/{linear,conv,rnn}.py` and `metric/{correlation,stability}.py` are 0 bytes.
This is therefore a from-scratch port; that memory is retained only as a record of
agreed conventions.

## Fidelity criterion (what "done" means per item)

Each ported item passes a test in `tests/`:
- **Streaming kernels & metrics:** encode -> run N timesteps -> decode; progressive
  error vs the analytic reference within a stochastic-computing bound (~1/sqrt(N)).
  Matches the existing `test_mul_gaines.py` -> `accuracy.analyze(reference)` pattern.
- **Binary-domain trainable kernels (HUB/FXP/TLUT/Hard):** single-shot whole-tensor
  `forward()` vs the `nn.Linear`/`nn.Conv2d` float reference within the quantization bound.
- **Known-answer sanity checks:** `mul_gaines` unipolar (1,1)->1,(1,0)->0; metric on a
  perfect stream reports ~0 error; Correlation of a stream with itself -> +1, with its
  inverse -> -1.

Validation is **napl-native** (no UnarySim runtime dependency; UnarySim is a read-only
reference for the math). **Python-only**: the `imp/` RTL tree is untouched.

## Gap analysis (UnarySim main -> napl)

Already present (no work): FSUMul->mul_gaines/mul_ugemm, FSUAdd->add_any, FSUDiv/CORDIV->div_*,
FSUReLU->relu_*, FSUSqrt->sqrt_*, FSUSignAbs->signabs, FSUHardsigmoid->sigmoid_hard,
FSUHardtanh->tanh_hard, JKFF/ShiftReg/Bi2Uni/Uni2Bi/SkewedSync, ProgError->metric/accuracy,
RNG/BinGen/BSGen/RawScale folded into encoder/decoder/gen_num_seq.

To port:
- **Metrics:** Correlation -> metric/correlation.py; Stability -> metric/stability.py
- **Round primitive:** Round/RoundSTE -> operation/round_fxp.py (round_ste, round_fxp)
- **Linear:** FSULinear(+PC) -> linear; HUB/FXP/TLUT -> linear_hub/_fxp/_tlut
- **Conv:** FSUConv2d(+PC) -> conv; HUB/FXP/TLUT -> conv_hub/_fxp/_tlut
- **RNN:** FSUMGUCell -> mgu; HUBMGUCell/HardMGUCell/HardMGUCellFXP -> mgu_hub/_hard/_hardfxp
- **Activations (binary):** HUBReLU/HUBHardsigmoid/HUBHardtanh -> *_hub
- **utils helpers:** conv2d padding/shape, num2tuple, rshift_offset, truncated_normal,
  NN_SC_Weight_Clipper, float-shift shims (pow2_lshift/pow2_rshift)

Out of scope: `structure/` (biological layer), `algorithm/fft/fft.py`.

## Conventions

- UnarySim hwcfg/swcfg dicts -> napl `config` dict (`polarity`, `timestep`, `generator`, `dim`).
- Variant as lowercase suffix (linear, linear_hub, ...).
- Every kernel is a `napl_base` subclass. Streaming = per-timestep forward() under
  `@napl_sim_timesteps` (streaming calls auto-tick); HUB/FXP/TLUT/Hard = single-shot (`streaming = False`, no tick), trainable via
  torch.autograd.Function + STE.
- Gotchas: no `>>`/`<<` on float tensors (use pow2_lshift/pow2_rshift shims);
  operation is self-contained and encoder/decoder live in it, so module.{linear,conv,rnn}
  import operation primitives lazily inside __init__ only as a leftover; conv bipolar zero-pad uses a
  decorrelated rate-0.5 pad stream.
- One test file per kernel under `tests/`, runnable by pytest and as a script;
  `tests/sweep_test.py` runs the full suite.

## Build milestones (each gated green before the next)

1. **[DONE]** Walking skeleton: metric/correlation + metric/stability + operation/round_fxp +
   linear, one test each, end-to-end green. SC-customized adversarial review passed
   (2 silent-corruption bugs found & fixed: accumulator-width saturation guard, LFSR
   decorrelation warning).
2. **[DONE]** linear_hub/_fxp/_tlut (all 3 TLUT modes) + relu_hub/sigmoid_hub/tanh_hub +
   utils helpers (rshift_offset, num2tuple, conv2d_output_shape/get_padding,
   truncated_normal, NN_SC_Weight_Clipper). Validated vs nn.Linear within quant bounds
   (fxp rmse 0.011, hub 0.026, tlut-fxpfxp 0.025), STE gradients flow. 42/42 pytest green.
   SC-customized B4 adversarial review passed (oracle = nn.Linear): 1 real bug found &
   fixed (all-zero input -> NaN via log2(0) in rshift_offset, guarded with nan_to_num +
   regression tests); STE-grad-exactness, sign-magnitude, matmul-axis, HUB-map invariants,
   quantization-active, config-threading, determinism all cleared.
3. **[DONE]** conv (streaming, im2col + decorrelated rate-0.5 bipolar pad stream) +
   conv_fxp/conv_hub/conv_tlut (binary, im2col + reuse of the M2 linear autograd Functions
   via a shared _conv2d_binary helper). HUB map builder extracted to _build_hub_map (shared
   with linear_hub). Validated vs nn.Conv2d: conv_fxp rmse 0.010, conv_hub 0.023, conv_tlut
   0.025; conv rmse 0.005 (pad 0 and 1), STE gradients flow, reset reproduces. 44/44 green.
   SC-customized B4 review passed CLEAN (0 findings, oracle = F.conv2d/nn.Conv2d): pad-stream
   proven load-bearing (encodes value 0 not -1), convergence ~1/sqrt(N), stride/dilation/rect
   correct, batch independence exact, binary STE grad == nn.Conv2d exactly, width guard fires,
   all-zero input finite (shares M2 rshift NaN guard).
4. **[DONE]** module/mgu.py, module/mgu_hard.py, module/mgu_hardfxp.py, and module/mgu_hub.py MGU cells: mgu_hard (binary float, == manual hard-MGU exactly),
   mgu_hardfxp (round_fxp everywhere, rmse 0.010 vs mgu_hard), mgu (streaming inner cell:
   linear scale=1 = linear+hardtanh, sigmoid_hard, mul_ugemm for fg*hx, mul_gaines for fg*ng,
   add_any scale=1 output), mgu_hub (hybrid: runs mgu over 2**width cycles, decodes with
   accuracy metric; rmse 0.032 vs mgu_hard). Caught & fixed a circular import (mgu.py must
   lazy-import napl.sim.operation inside __init__, like linear) that pytest masked but the
   script sweep exposed. 46/46 green, 0 sweep failures.
5. **[DONE]** Full integration sweep + final B4 review. 46/46 pytest, 0 script-sweep failures,
   all 5 package entry points import clean (no cycles regardless of load order). M4 RNN B4
   review passed CLEAN (oracle = manual MGU / mgu_hard): recurrence rollout exact over 8 steps,
   streaming unbiased (|mean err| 0.013 << rmse 0.074), width convergence, STE grads to both
   gate weights, fxp quantization active, bias=False path, determinism, outputs in [-1,1].
   One test bound corrected to be seed-robust (mgu_hub worst-case rmse ~0.11 at width 8).

PORT COMPLETE: all in-scope UnarySim metrics & kernels ported, each individually validated
and adversarially reviewed. Out of scope (unchanged): structure/, algorithm/fft/fft, the CLI.

## Top risk

Silent numerical infidelity (a port that runs but diverges from UnarySim under some
polarity/width). Mitigated by the fidelity criterion + known-answer checks + B4 review.
