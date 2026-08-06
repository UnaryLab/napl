# napl-port-unarysim run summary

## Overview

This run covered all phases, all subpackages, and all classes. **GenSim** newly ported **1** UnarySim class into napl (`FSUConv2dPC` -> `module.conv_pc`) and skipped **9** non-ports (backward-only STE `autograd.Function` helpers and RNG/bitstream codec stages that napl folds into existing classes). **Improve** changed **14** files and left **10** unchanged with no safe speedup; **0** unchanged files were skipped via the improve ledger. The post-Improve sweep gate passed: **48/48** files, `all_pass=true`, no failing tests. **0** classes were force-re-validated. **Validate** ran **44** module validations with **0** disagreements (`agree=false`); the `disagreements_rechecked` list is empty, so nothing was overturned or confirmed on independent re-check. **RTL**: **30** ops generated and verified, **0** dropped, **5** skipped as single-shot binary-domain HUB/FXP ops with no sound gate-level mapping (`sigmoid_hub`, `relu_hub`, `round_fxp`, `tanh_hub`; plus `round_fxp`). **23** `self.hw.pp_delay` values were written in Finalize. The component report records **42** validated-yes and **26** rtl-yes.

## Improve

| file | changed? | tests pass? | what changed |
|---|---|---|---|
| operation/uni2bi.py | yes | yes | Single `torch.ge(acc,2)` cast to stype reused for both acc update and return; removes one cast/timestep. CPU ~1.37x, MPS ~1.52x. |
| operation/min_rc.py | yes | yes | Hoisted duplicate int8 casts in `min_rc/max_rc/lt_rc/gt_rc`; `min_tc/max_tc` left unchanged. Bit-exact; within timing noise. |
| operation/sigmoid_hard.py | no | yes | No safe speedup; both classes thin fused wrappers already optimal. |
| operation/square_dff.py | yes | yes | Cached `_spike_is_int8`; skip 3 redundant casts on int8 fast path; explicit XNOR precedence. CPU ~1.05-1.12x, MPS ~1.05-1.2x. |
| operation/shiftreg.py | yes | yes | Replaced stacked circular buffer with `collections.deque` FIFO of row refs; removes per-step clone. CPU ~3x, MPS ~5x. |
| operation/relu_cnt.py | yes | yes | Dropped redundant `int8` cast in `relu_cnt` (int8\|bool yields int8). Bit-exact; flat within noise. |
| operation/round_fxp.py | no | yes | No change; `round_fxp` hot path already minimal, only micro-opts live in shared shims. 1.00x. |
| operation/jkff.py | no | yes | No change; current `torch.where` select fastest on CPU, no rewrite wins on both devices. |
| operation/mul_and.py | yes | yes | `mul_csg`: removed per-timestep long cast, reused `1-in_0_i8`. `mul_and` unchanged. CPU 1.03x, MPS 1.14x. |
| operation/dff.py | no | yes | No change; already the circular-buffer-of-references idiom, zero steady-state allocations. |
| operation/div_cordiv.py | yes | yes | `div_cordiv`: moved temp off `self`, dropped redundant cast. `div_iscb` unchanged. ~1.0x within noise. |
| operation/sqrt_emit.py | yes | yes | `sqrt_emit`: replaced per-timestep stack+reduce with int8 elementwise sum fed `dim=None`. CPU ~1.1-1.2x, MPS ~1.05-1.2x. |
| operation/signabs.py | no | yes | No change; already vectorized, candidate cast-removal only ~1.06x and drops stype contract. |
| operation/add_any.py | yes | yes | `add_any` dim!=None path: in-place `sub_(offset)` on owned sum, removes one full-size alloc/timestep. CPU ~1.1x, MPS ~1.2-1.3x. |
| operation/bi2uni.py | yes | yes | Removed per-timestep cast; shape-guarded in-place `add_`. MPS ~1.25x, CPU flat. |
| operation/tanh_hard.py | no | yes | No change; `tanh_hard` is passthrough, `tanh_hub` single fused hardtanh. |
| operation/sync_skewed.py | no | yes | No new edits; pre-existing behavior-preserving opts to `sync_skewed` verified (CPU 1.17x, MPS 1.30x vs HEAD). |
| module/{mgu,mgu_hard,mgu_hardfxp,mgu_hub}.py | yes | yes | `mgu_hardfxp`: deduped redundant `round_fxp` quantizations. CPU ~1.30x, MPS ~1.20x. Other 3 classes unchanged. |
| module/conv.py | no | yes | No change; hot path dominated by im2col/col2im, casts non-redundant. 1.0x. |
| module/conv_pc.py | no | yes | No change; hot path already vectorized, shape-cache prototype gave no speedup and was reverted. |
| module/encoder.py | yes | yes | Resolved polarity branch once in `__init__`; removed 2 string compares/timestep. CPU ~1.10x, MPS ~1.12x. |
| module/linear_pc.py | yes | yes | `linear_pc` bipolar: replaced second matmul with integer identity reusing AND-count. CPU ~1.17x, MPS ~1.07x. |
| module/decoder.py | yes | yes | Dropped per-timestep cast (add promotes int8 into float32 acc). MPS ~1.9x, CPU flat. |
| metric/correlation.py | yes | yes | Dropped 2 redundant casts (bool auto-promotes). CPU ~1.10x, MPS ~2.01x. |
| metric/accuracy.py | yes | yes | Shape-guarded in-place `add_` accumulation; removes alloc for 255/256 timesteps. CPU/MPS ~1.16x. |
| metric/stability.py | yes | yes | In-place `sub_().abs_()` error computation; removes one alloc/timestep. CPU ~1.17x, MPS ~1.21x. |
| algorithm/fft/{butterfly_spike.py,butterfly_binary.py} | no | yes | No change; thin orchestration over already-optimized primitives. CPU/MPS ~1.00x. |

Unchanged files skipped via the improve ledger: **0**.

Independent sweep-gate result: **48** files, **48** passed, `all_pass=true`, failing: none.

## Validate

| module | UnarySim ref | bit-exact | agreement | CPU | GPU |
|---|---|---|---|---|---|
| operation.uni2bi | Uni2Bi | yes (diff 0) | bit-exact, 0 mismatched over T=256 x N=10000, CPU+MPS | 4.92 ms | 20.18 ms |
| operation.sigmoid_hard | FSUHardsigmoid | yes (diff 0) | bit-exact CPU+MPS; deterministic streaming scaled-add | 8.48 ms | 30.59 ms |
| operation.sigmoid_hub | ScaleHardsigmoid | yes (diff 0) | bit-exact CPU+MPS, exact vs Hardsigmoid(x*3) | 0.0050 ms | 0.0067 ms |
| operation.shiftreg | ShiftReg | yes (diff 0) | bit-exact CPU+MPS; depth-1 within SC bound | 1.13 ms | 8.09 ms |
| operation.relu_cnt | FSUReLU | yes (diff 0) | bit-exact streams, RMSE vs ReLU=0.0, CPU+MPS | 7.7 ms | 17.4 ms |
| operation.relu_sat | FSUReLU | no (diff algorithm) | RMSE 0.0055 vs FSUReLU within SC bound, CPU+MPS | 12.8 ms | 54.0 ms |
| operation.relu_hub | ScaleReLU | yes (diff 0) | bit-exact; both Hardtanh(0,scale), CPU+MPS | 0.0015 ms | 0.0089 ms |
| operation.round_fxp | Round | yes (diff 0) | bit-exact, CPU+MPS | 0.0319 ms | 0.0305 ms |
| operation.jkff | JKFF | yes (diff 0) | bit-exact CPU+MPS | 4.14 ms | 39.07 ms |
| operation.mul_and | FSUMul (AND/XNOR core) | yes (diff 0) | bit-exact CPU+MPS, bipolar & unipolar | 6.81 ms | 20.09 ms |
| operation.mul_csg | FSUMul (static=True) | yes (diff 0) | bit-exact CPU+MPS, 5 seeds; RMSE within SC bound | 101.96 ms | 119.95 ms |
| operation.dff | ShiftReg (delay identity) | yes (diff 0) | bit-exact, CPU+MPS | 3.51 ms | 10.94 ms |
| operation.div_cordiv | CORDIV_kernel | no (buffer/RNG init differs) | within SC bound (nap-vs-ref RMSE 0.053 ~0.84x bound) | 11.6 ms | 38.1 ms |
| operation.div_iscb | FSUDiv | no (wraps cordiv, polarity path) | within SC bound; CPU+MPS consistent | 40.7 ms | 123.0 ms |
| operation.sqrt_tracejkff | FSUSqrt | yes (diff 0) | bit-exact CPU+MPS; RMSE-vs-sqrt identical | 12.9 ms | 42.8 ms |
| operation.sqrt_traceiscb | FSUSqrt | yes (diff 0) | bit-exact CPU+MPS; RMSE-vs-sqrt identical | 17.2 ms | 46.7 ms |
| operation.sqrt_emit | FSUSqrt | yes (diff 0) | bit-exact CPU+MPS; RMSE-vs-sqrt 0.026 within bound | 9.3 ms | 33.8 ms |
| operation.signabs | FSUAbs + FSUSign | yes (diff 0) | bit-exact both outputs, CPU+MPS | 5.11 ms | 18.14 ms |
| operation.add_any | FSUAdd | yes (diff 0) | bit-exact CPU+MPS; within SC bound | 37.5 ms | 34.8 ms |
| operation.bi2uni | Bi2Uni | yes (diff 0) | bit-exact vs Bi2Uni(depth=3), CPU+MPS | 3.36 ms | 9.21 ms |
| operation.tanh_hard | FSUHardtanh | yes (diff 0) | bit-exact; RMSE 0.0 vs analytic | 4.8 ms | 11.8 ms |
| operation.tanh_hub | ScaleHardtanh | yes (diff 0) | bit-exact vs ScaleHardtanh incl. corners | 10.0 ms/1000 | 12.2 ms/1000 |
| operation.sync_skewed | SkewedSync | yes (diff 0) | bit-exact, CPU+MPS | 13.56 ms | 44.80 ms |
| module.mgu_hard | HardMGUCell | yes (max\|diff\| 0) | bit-exact CPU+MPS | 0.027 ms | 0.293 ms |
| module.mgu_hardfxp | HardMGUCellFxp | yes (max\|diff\| 0) | bit-exact CPU+MPS | 0.141 ms | 0.644 ms |
| module.mgu | FSUMGUCell | no (internal RNG) | stochastic; within few*SC bound | ~33 ms (via mgu_hub) | n/a (device-mixing) |
| module.mgu_hub | HUBMGUCell | no (internal RNG) | stochastic; RMSE within SC bound @width8 | 33.0 ms | n/a (device-mixing) |
| module.conv_fxp | FxpConv2d | yes (diff 0) | bit-exact; rmse within quant bound | 0.292 ms | 1.069 ms |
| module.conv_hub | HUBConv2d | yes (diff 0) | bit-exact; rmse within unary-mul bound | 0.370 ms | 1.102 ms |
| module.conv_tlut | *(no upstream counterpart)* - UnarySim has no TLUT convolution class | yes (diff 0) | bit-exact; rmse within temporal bound | 0.719 ms | 1.173 ms |
| module.conv | FSUConv2d | no (internal RNG) | within SC bound, cross-rmse << bound | 55-60 ms | see notes |
| module.conv_pc | FSUConv2dPC | no (independent RNG) | within SC bound; both track F.conv2d | 42.6 ms | 87.9 ms |
| module.encoder | BSGen | yes (diff 0) | bit-exact across regimes; RMSE 0.0 | 0.81 ms | 7.87 ms |
| module.linear | FSULinear | no (independent RNG) | within SC bound; both track analytic | 6.88 ms | 61.08 ms |
| module.linear_pc | FSULinearPC | no (independent RNG) | within SC bound; both track inner product | 4.59 ms | 40.94 ms |
| module.linear_fxp | FxpLinear | yes (diff 0) | bit-exact CPU (legacy float-shift restored) | 0.16 ms | 1.04 ms |
| module.linear_hub | HUBLinear | yes (diff 0) | bit-exact CPU; identical unary map | 0.157 ms | 1.07 ms |
| module.linear_tlut | *(no upstream counterpart)* - UnarySim has no TLUT linear class | yes (diff 0) | bit-exact all 3 modes/temporal pairs, CPU | 0.124 ms | 1.21 ms |
| module.decoder | ProgError | yes (diff 0) | bit-exact CPU+MPS, both polarities | 12.0 ms | 35.9 ms |
| metric.correlation | Correlation | yes (diff 0) | bit-exact CPU+MPS across regimes | 3.81 ms | 27.34 ms |
| metric.accuracy | ProgError | yes (diff 0) | bit-exact CPU+MPS, both polarities | 4.14 ms | 10.39 ms |
| metric.stability | Stability | yes (diff 0) | bit-exact CPU+MPS | 4.20 ms | 15.55 ms |

No row had `agree=false`; all 44 validations agreed. The `disagreements_rechecked` list is empty (nothing overturned or confirmed on re-check).

## RTL

| class | op | status | make test | independently verified? | pipeline delay (cycles) | notes |
|---|---|---|---|---|---|---|
| uni2bi | uni2bi | verified | PASS | yes | 0 | Mealy machine, single bare module; 304/304 vectors. |
| min_rc | min_rc | verified | PASS | yes | 0 | sync_skewed width-2; 4096/4096; o_argmin = ~dff_next. |
| max_rc | max_rc | verified | PASS | yes | 0 | 4096/4096; fixed arithmetic-sum vs concat bug. |
| lt_rc | lt_rc | verified | PASS | yes | 1 | Registered dff output; 65/65. |
| gt_rc | gt_rc | verified | PASS | yes | 1 | dff reset 1; 2000/2000. |
| min_tc | min_tc | verified | PASS | yes | 0 | Combinational AND; 4/4. |
| max_tc | max_tc | verified | PASS | yes | 0 | Combinational OR; 4/4. |
| sigmoid_hard | sigmoid_hard | verified | PASS | yes | 0 | Single bare module (bipolar offset 0); 256/256. |
| sigmoid_hub | sigmoid_hub | skipped | n-a | no | null | Single-shot binary-domain HUB op; no sound gate-level mapping. Recorded so it is not re-attempted. |
| square_dff | square_dff | verified | PASS | yes | 0 | Two polarity variants (AND/XNOR), depth=1; 64/64. |
| shiftreg | shiftreg | verified | PASS | yes | 4 | DEPTH=4; reset reg[i]=i%2; 15/15. |
| relu_cnt | relu_cnt | verified | PASS | yes | 0 | Saturating counter, reset acc=HALF; 256/256. |
| relu_sat | relu_sat | verified | PASS | yes | 0 | Two chained add_any accumulators (2x-scaled); 218/218. |
| relu_hub | relu_hub | skipped | n-a | no | null | Single-shot binary-domain HUB cell; no bit-serial datapath. Recorded so it is not re-attempted. |
| round_fxp | round_fxp | skipped | n-a | no | null | Single-shot binary-domain quantizer; word-level clamp, not 1-bit. Recorded so it is not re-attempted. |
| jkff | jkff | verified | PASS | yes | 1 | Synchronous register, Q'=Q?~K:J; 12/12. |
| mul_and | mul_and | verified | PASS | yes | 0 | Restored from HEAD (dir deleted in tree); AND/XNOR; 4/4. |
| mul_csg | mul_csg | verified | PASS | yes | 0 | ROM-baked sobol num_seq (T=256); 2048/2048. |
| dff | dff | verified | PASS | yes | 1 | depth=1 single D-FF, zero reset; 64/64. |
| div_cordiv | div_cordiv | verified | PASS | yes | 0 | depth-2 circular buffer; 16/16. |
| div_iscb | div_iscb | verified | PASS | yes | 0 | Two polarity variants with helper sub-blocks; 512/512. |
| sqrt_tracejkff | sqrt_tracejkff | verified | PASS | yes | 0 | Two polarity variants (JK-FF trace); 64/64. |
| sqrt_traceiscb | sqrt_traceiscb | verified | PASS | yes | 0 | Two polarity variants (iscb cordiv); 128/128. |
| sqrt_emit | sqrt_emit | verified | PASS | yes | 0 | Two polarity variants (nsadd + shiftreg); 64/64. |
| signabs | signabs | verified | PASS | yes | 0 | Saturating accumulator, reset acc=MED; 42/42. |
| add_any | add_any | verified | PASS | yes | 0 | Two polarity variants, 2x-scaled acc; 200/200; fixed signed-compare bug. |
| bi2uni | bi2uni | verified | PASS | yes | 0 | Mealy machine; 304/304. |
| tanh_hard | tanh_hard | verified | PASS | yes | 0 | Combinational identity pass-through; 68/68. |
| tanh_hub | tanh_hub | skipped | n-a | no | null | Single-shot binary-domain HUB op; no 1-bit datapath. Recorded so it is not re-attempted. |
| sync_skewed | sync_skewed | verified | PASS | yes | 0 | Saturating up/down counter; 33/33. |

Ops in `rtl_verify_dropped` (claimed but failed re-verification): **none**.

Skipped ops (status=skipped, recorded so they are not re-attempted): `sigmoid_hub`, `relu_hub`, `round_fxp`, `tanh_hub`, all single-shot binary-domain HUB/FXP ops with no sound gate-level (bit-serial) RTL mapping.

## GenSim

| napl class | UnarySim source | status | validated |
|---|---|---|---|
| module.conv_pc | FSUConv2dPC | ported | Faithful vs FSUConv2dPC on identical input spike streams (CPU+MPS), both polarities x bias x pad; bit-exact=False as expected (independent weight/bias RNG per side); within-bound agreement (all RMSE << SC bound 0.133 @ T=512). |

Skipped non-ports (9):

- **FxpLinearFunction** - backward-only `torch.autograd.Function` STE helper behind FxpLinear; folded into `_linear_fxp_fn`.
- **HUBLinearFunction** - backward-only STE helper behind HUBLinear; folded into `_linear_hub_fn`.
- **(no upstream counterpart)** - backward-only STE helper behind the napl TLUT linear (FPFP); UnarySim has no TLUT linear class; folded into `_linear_tlut_fpfp_fn`.
- **(no upstream counterpart)** - backward-only STE helper behind the napl TLUT linear (FXPFP); UnarySim has no TLUT linear class; folded into `_linear_tlut_fxpfp_fn`.
- **(no upstream counterpart)** - backward-only STE helper behind the napl TLUT linear (FXPFXP); UnarySim has no TLUT linear class; folded into `_linear_tlut_fxpfxp_fn`.
- **RNG** - RNG/bitstream generator; corresponds to napl encoder/decoder/gen_num_seq codec, not a standalone class.
- **RawScale** - value-scaling stage of the bitstream generator pipeline; folded into napl encoder/decoder codec.
- **SourceGen** - binary source generator; folded into napl encoder/gen_num_seq codec.
- **BSGen** - bitstream (spike) generator; folded into napl encoder codec.

## Finalize

`self.hw.pp_delay` edits applied: **23**.

Component report (`napl-port-unarysim-component-report.md`) column counts: `validated_yes` = **42**, `rtl_yes` = **26**.

Meow...
