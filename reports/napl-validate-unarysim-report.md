# napl vs UnarySim validation log

Records produced by the `napl-validate-unarysim` skill - one row per validation run. **Bit-exact**
means `torch.equal` held (max abs diff 0): deterministic ports (metrics, gate ops) should be exact,
while RNG-driven streaming kernels (`linear`, `conv`) only agree within the
stochastic-computing bound ~1/sqrt(N) and are recorded as not bit-exact with their agreement RMSE.
**Runtimes** are the napl module's wall-clock on CPU vs GPU (MPS) for the noted workload (GPU timed
with `torch.mps.synchronize()`); they are per-workload, not comparable across rows.

| Date | napl module | UnarySim ref | Bit-exact | Agreement | CPU runtime | GPU (MPS) runtime | Regimes / workload |
|------|-------------|--------------|-----------|-----------|-------------|-------------------|--------------------|
| 2026-06-13 | metric.accuracy | ProgError | yes (diff 0) | bit-exact; max\|val_diff\|=0 and max\|err_diff\|=0 on CPU+MPS, both polarities | 4.14 ms | 10.39 ms | bipolar, unipolar; T=256, N=10000; known-answer (pp converges to source, rmse_vs_ref=0); spike_value==pp, spike_error==pe |
| 2026-07-22 | metric.accuracy | ProgError | yes (value, err, per-timestep) | max\|diff\|=0 all regimes, CPU+MPS | 2.19 ms | 5.15 ms | bipolar, unipolar; N=10000 T=256 via mul_csg test wiring |
| 2026-06-13 | metric.correlation | Correlation | yes (diff 0) | bit-exact (max\|diff\|=0) on CPU and MPS across all regimes | 3.81 ms | 27.34 ms | bipolar; SCC of stream vs identical (+1), vs complement (-1), vs independent dim (~0), and single-input autocorrelation; N=1000, T=256 |
| 2026-07-22 | metric.correlation | Correlation | yes (CPU+MPS, all regimes) | max\|diff\|=0; napl 1.42x faster CPU, 1.35x faster MPS vs UnarySim | 2.31 ms | 13.95 ms | bipolar N=1000 T=256; self, complement, independent, autocorr |
| 2026-06-13 | metric.stability | Stability | yes (diff 0) | bit-exact, max\|diff\|=0 on CPU+MPS | 4.20 ms | 15.55 ms | bipolar & unipolar; thresholds 0.01/0.05/0.20; zero-source edge; N=1000 T=256; deterministic (no internal RNG) |
| 2026-07-22 | metric.stability | Stability | yes (CPU+MPS, bipolar+unipolar) | max\|diff\|=0, deterministic function of same spikes | 2.95 ms | 9.40 ms | bipolar, unipolar; N=1000 T=256 thr=0.05 sobol |
| 2026-07-22 | metric.stability_builder | NSbuilder | yes (per-step, CPU and MPS) | all 256 steps identical, stream-count max\|diff\|=0 | 2.8 ms | 20.2 ms | bipolar+unipolar, normstability 0.1/0.5/0.9; N=1000 T=256 thr=0.05 |
| 2026-06-13 | module.conv | FSUConv2d | no (internal weight RNG on both sides) | not bit-exact (each side streams weights with its own RNG); napl rmse 0.0048/0.0049 vs analytic, UnarySim 0.0044/0.0043, cross-rmse 0.0037-0.0039, all << SC bound 0.0625 (T=256) | 55-60 ms (256-step sweep) | see notes | bipolar streaming, scaled-add (scale=entry); pad 0 and 1; b=2 ic=3 oc=4 hw=8 k=3 T=256 width=12 Sobol |
| 2026-06-13 | module.conv_pc | FSUConv2dPC | no (independent weight bitstream RNG: napl encoder vs UnarySim BinGen/RNG/BSGen) | napl vs UnarySim RMSE 0.0195-0.0263 (max\|diff\| <=0.094) vs SC bound 0.044; both track F.conv2d within bound (napl RMSE 0.005-0.024, uref RMSE 0.010-0.022); CPU+MPS identical | 42.6 ms | 87.9 ms | unipolar+bipolar x bias{T,F} x pad{0,1}; B=2 in=3 out=4 HW=8 k=3 T=512; same input spike stream fed to both, each its own weight RNG; known-answer all-ones unipolar count==entry |
| 2026-06-13 | module.conv_fxp | FXPConv2d | yes (diff 0) | bit-exact vs UnarySim, max\|diff\|=0 (pad 0,1); rmse 0.012-0.013 vs nn.Conv2d within quant bound | 0.292 ms | 1.069 ms | binary-domain single-shot; identical weight/input/bias; pad 0 and 1; b=4 ic=3 oc=6 hw=10 k=3 w8 |
| 2026-06-13 | module.conv_hub | HUBConv2d | yes (diff 0) | bit-exact vs UnarySim, max\|diff\|=0 (pad 0,1); rmse 0.023-0.024 vs nn.Conv2d within unary-mul bound | 0.370 ms | 1.102 ms | binary-domain single-shot; identical weight/input/bias; pad 0 and 1; widthi=widthw=8 Sobol cycle=128 signmag |
| 2026-06-13 | module.conv_tlut | TLUTConv2d | yes (diff 0) | bit-exact vs UnarySim, max\|diff\|=0 (pad 0,1); rmse 0.014-0.015 vs nn.Conv2d within temporal-decomp bound | 0.719 ms | 1.173 ms | binary-domain single-shot; fxpfxp mode temporal=i widtht=4 widthi=widthw=8; pad 0 and 1 |
| 2026-06-13 | module.decoder | ProgError | yes (diff 0) | bit-exact (max\|diff\|=0 on CPU and MPS, both polarities) | 12.0 ms | 35.9 ms | bipolar + unipolar; identical Sobol-encoded streams T=1024 N=10000; known-answer all-ones (->1.0) / all-zeros (->0.0 uni, -1.0 bip) |
| 2026-06-13 | module.encoder | BSGen | yes (diff 0) | bit-exact (torch.equal=True, max\|diff\|=0) across regimes; RMSE 0.0 vs SC bound 0.031-0.0625 | 0.81 ms | 7.87 ms | bipolar+unipolar; T=256/512/1024; Sobol dim=1/2; shapes (100,)/(500,)/(1000,); validated against UnarySim RNG+BinGen+BSGen pipeline (mapping: encoder = RNG+RawScale+BinGen+BSGen) |
| 2026-06-13 | module.linear | FSULinear | no (independent weight RNG) | napl-vs-UnarySim RMSE 0.0048 (bipolar) / 0.0024 (unipolar) vs SC bound ~0.0625; both track analytic ref (napl RMSE 0.0056/0.0023, UnarySim 0.0025/0.0009); CPU only for ref | 6.88 ms | 61.08 ms | bipolar, unipolar; in=16 out=8 bias=True T=256 width=12; analytic (Wx+b)/(in+1) reference, sobol input dim1 / weight dim2 |
| 2026-06-13 | module.linear_pc | FSULinearPC | no (independent weight RNG) | napl-vs-UnarySim RMSE 0.0072 (bipolar) / 0.0018 (unipolar) vs SC bound ~0.0312; both track analytic inner product (napl 0.0069/0.0017, UnarySim 0.0040/0.0010); CPU only for ref | 4.59 ms | 40.94 ms | bipolar, unipolar; in=16 out=8 bias=True T=1024; per-timestep PC count accumulated/T -> Wx+b, AND-count (uni) / XNOR-count (bi) |
| 2026-06-13 | module.linear_fxp | FXPLinear | yes (diff 0) | bit-exact, max\|diff\|=0 on CPU (UnarySim ref needs legacy float-shift restored, matching napl pow2_rshift); deterministic same-input fxp matmul | 0.16 ms | 1.04 ms | shared weight/bias/input, batch=8 in=32 out=16; widthi=widthw=8 quantile=1 rounding=round; STE backward |
| 2026-06-13 | module.linear_hub | HUBLinear | yes (diff 0) | bit-exact, max\|diff\|=0 on CPU; identical unary value map (mapcbsg) and sign-magnitude lookup, legacy float-shift restored for UnarySim ref | 0.157 ms | 1.07 ms | shared weight/bias/input, batch=8 in=32 out=16; widthi=widthw=8 rngi=rngw=sobol cycle=128 quantile=1 round signmag |
| 2026-06-13 | module.linear_tlut | TLUTLinear | yes (diff 0) | bit-exact, max\|diff\|=0 across all 3 modes/temporal pairs (fxpfxp temporal=i and =w, fxpfp, fpfp) on CPU; legacy float-shift restored for UnarySim ref | 0.124 ms | 1.21 ms | shared weight/bias/input, batch=8 in=32 out=16; modes fxpfxp(i/w)/fxpfp/fpfp, widtht=4 widthi=widthw=8 round signmag; STE backward |
| 2026-06-13 | module.mgu | FSUMGUCell | no (internal weight/mul RNG) | stochastic; nap-vs-uhub RMSE 0.07-0.085 across seeds, both within few*SC bound (~0.0625 @ width8) of float MGU; exercised inside mgu_hub | ~33 ms (via mgu_hub, width=8) | n/a (device-mixing in streaming path on MPS) | bipolar rate-coded streaming inner cell; width=8 (256 cycles); isz=6 hsz=4 b=3; independent decorrelated Sobol encoders vs UnarySim CSG/in-stream weight indexing |
| 2026-06-13 | module.mgu_hard | HardMGUCell | yes (max\|diff\| 0) | bit-exact CPU+MPS; max\|diff\|=0 | 0.027 ms | 0.293 ms | bipolar hard activations; isz=6 hsz=4 b=3 bias=True; shared weights+inputs; deterministic single-shot |
| 2026-06-13 | module.mgu_hardfxp | HardMGUCellFXP | yes (max\|diff\| 0) | bit-exact CPU+MPS; max\|diff\|=0 (intwidth=3,fracwidth=4; round_fxp == Round STE) | 0.141 ms | 0.644 ms | bipolar hard+fxp quant intwidth=3 fracwidth=4; isz=6 hsz=4 b=3; shared weights+inputs; UnarySim Round needs RAVEN float-shift shim to run |
| 2026-06-13 | module.mgu_hub | HUBMGUCell | no (internal input/hx/weight RNG) | stochastic; nap-vs-uhub RMSE 0.074 (seed0), 0.07-0.085 across seeds; nap-vs-float 0.064, uhub-vs-float 0.050; SC bound ~0.0625 @ width8 | 33.0 ms | n/a (device-mixing in streaming path on MPS) | bipolar hybrid, Sobol, width=8 (256 cycles); isz=6 hsz=4 b=3 bias=True; quantized int8 spike inputs; shared weights; runs inner mgu, decodes via accuracy metric |
| 2026-06-13 | operation.add_any | FSUAdd | yes (diff 0) | bit-exact on CPU+MPS (maxdiff=0); both track analytic ref within SC bound (RMSE ~0.0045 bipolar / ~0.0022 unipolar vs bound 0.0625) | 37.5 ms | 34.8 ms | bipolar & unipolar; deterministic (no internal RNG); scale=128 entry=128 width=20 T=256 n=2000; reduce dim=-1; analytic known-answer (sum/scale) |
| 2026-06-13 | operation.bi2uni | Bi2Uni | yes (diff 0) | bit-exact (max\|diff\|=0) vs Bi2Uni(depth=3) on CPU and MPS; deterministic gate op, no internal RNG | 3.36 ms | 9.21 ms | bipolar sobol dim=1 stream, width=2 <-> depth=3; T=256 N=10000; plus all-ones / all-zero / rand-p0.5 adversarial streams. width=2 maps to UnarySim depth=width+1 (matches depth=3, diverges from depth=2 by parameter choice, not algorithm). |
| 2026-06-13 | operation.dff | ShiftReg (delay identity; no standalone DFF kernel) | yes (diff 0) | bit-exact, CPU+MPS | 3.51 ms | 10.94 ms | bipolar; depth=1 and depth=3; vs analytic delay-identity and UnarySim ShiftReg(zero-init, index=0); T=256/64, shape=(10,)/(5,) |
| 2026-06-13 | operation.div_cordiv | CORDIV_kernel | no (internal buffer/RNG init differs: napl buffer_q init 0 + gen_num_seq index vs UnarySim ShiftReg sr init [i%2] + historic_q init 1 + RNG dimr=4, opposite roll direction) | agree within SC bound: napl RMSE 0.0258 vs analytic, ref RMSE 0.0367 vs analytic, nap-vs-ref RMSE 0.053 (~0.84x SC bound 0.0625); CPU+MPS consistent | 11.6 ms | 38.1 ms | unipolar; pre-synced dividend<divisor, divisor!=0; N=10000 T=256 depth=2 |
| 2026-06-13 | operation.div_iscb | FSUDiv | no (wraps div_cordiv/CORDIV_kernel whose internal RNG/buffer init differs; plus signabs + bi2uni/uni2bi polarity path) | agree within SC bound: napl RMSE 0.0589 vs analytic, ref RMSE 0.0648 vs analytic (SC bound 0.0625), nap-vs-ref RMSE 0.037 (~0.6x bound); MPS worst-case single element 0.99 near divide-by-small edge but aggregate within bound; CPU+MPS consistent | 40.7 ms | 123.0 ms | bipolar; \|dividend\|<\|divisor\|, divisor!=0; N=10000 T=256 |
| 2026-06-13 | operation.jkff | JKFF | yes (diff 0) | bit-exact on CPU+MPS, max\|diff\|=0.0 | 4.14 ms | 39.07 ms | all 4 JK states (set/reset/hold/toggle) known-answer; T=512 x N=1000 random 0/1 JK spike streams; CPU+MPS |
| 2026-06-13 | operation.mul_and | FSUMul (AND/XNOR gate core; no standalone UnarySim module) | yes (diff 0) | bit-exact CPU+MPS, bipolar & unipolar (max\|diff\|=0) | 6.81 ms | 20.09 ms | bipolar (Sobol dim1 x dim2) & unipolar; pure gate of two decorrelated externally-encoded streams, no RNG; N=10000 T=256 |
| 2026-06-13 | operation.mul_csg | FSUMul (static=True) | yes (diff 0) | bit-exact CPU+MPS, bipolar & unipolar across 5 seeds (decoded-value diff=0, per-step max\|diff\|=0); napl & ref RMSE vs analytic identical (bipolar 0.0024, unipolar 0.00065) within SC bound 0.03125 | 101.96 ms | 119.95 ms | bipolar & unipolar; streamed in_0 x static value in_1 with conditional spike generation; N=10000 T=1024 (5 seeds) |
| 2026-06-13 | operation.relu_cnt | FSUReLU | yes (diff 0) | bit-exact spike streams (per-step max\|diff\|=0, spike-sum identical), RMSE vs ReLU=0.0000, CPU+MPS | 7.7 ms | 17.4 ms | bipolar rate-coded; depth=width=3; shape=10000, T=256; vs analytic ReLU and FSUReLU on identical encoded streams |
| 2026-06-13 | operation.relu_hub | HUBReLU | yes (diff 0) | bit-exact (max\|diff\|=0); both are Hardtanh(0, scale); CPU+MPS | 0.0015 ms | 0.0089 ms | binary-domain single-shot; scale=1.0; input 4x128 spanning <0, [0,scale], >scale |
| 2026-06-13 | operation.relu_sat | FSUReLU | no (different algorithm: saturating-adder vs counter) | RMSE 0.0055 vs FSUReLU (SC bound ~0.0625); both track analytic ReLU; CPU+MPS | 12.8 ms | 54.0 ms | bipolar rate-coded; shape=10000, T=256; vs FSUReLU(depth=3) and analytic ReLU on identical encoded streams |
| 2026-06-13 | operation.round_fxp | Round | yes (diff 0) | bit-exact, max\|diff\|=0 on CPU+MPS | 0.0319 ms | 0.0305 ms | single-shot binary-domain quantizer; cfg (intwidth,fracwidth) = (3,4),(1,7),(5,2); 10011 inputs incl. random spread, exact-grid values, clamp corners (+/-1e6, +/-100), zeros, negatives |
| 2026-06-13 | operation.shiftreg | ShiftReg | yes (diff 0) | bit-exact, CPU+MPS; depth-1 delay decodes to input within SC bound (max err 0.0078 vs ~0.0625) | 1.13 ms | 8.09 ms | bipolar+unipolar; depth 1/2/4; shape=(10,) T=256; vs ShiftReg(index=0, mask=None); known-answer depth-1 unit delay |
| 2026-06-13 | operation.sigmoid_hard | FSUHardsigmoid | yes (diff 0) | bit-exact on CPU+MPS (max\|diff\|=0, identical spike counts); deterministic streaming scaled-add (x+1)/2, no internal RNG | 8.48 ms | 30.59 ms | bipolar; N=10000 random bipolar inputs, T=256; same Sobol-dim1 spike stream fed to both via mirror encoders, compared as bipolar-decoded value |
| 2026-06-13 | operation.sigmoid_hub | HUBHardsigmoid | yes (diff 0) | bit-exact on CPU+MPS (max\|diff\|=0) and exact vs Hardsigmoid(x*3) math; single-shot binary-domain, no RNG | 0.0050 ms | 0.0067 ms | binary-domain single-shot, scale=3; x=linspace(-4,4,20001) covering below -1 / linear band / above +1 |
| 2026-06-13 | operation.signabs | FSUSignAbs | yes (diff 0) | bit-exact both outputs (sign & abs) on CPU+MPS, max\|diff\|=0 | 5.11 ms | 18.14 ms | bipolar (rate coding only); width/depth=3; T=256; N=10000; deterministic per-timestep sign+abs |
| 2026-06-13 | operation.sqrt_emit | FSUSqrt | yes (diff 0) | bit-exact CPU+MPS; spike_diff=0 over T=256, N=4000; both RMSE-vs-sqrt=0.026 (within SC bound 0.0625) | 9.3 ms | 33.8 ms | unipolar; FSUSqrt(emit=True); shiftreg+nsadd emit path; identical encoded input stream; T=256, N=4000 |
| 2026-06-13 | operation.sqrt_traceiscb | FSUSqrt | yes (diff 0) | bit-exact CPU+MPS; spike_diff=0 over T=256, N=4000; napl & UnarySim RMSE-vs-sqrt identical (0.267) | 17.2 ms | 46.7 ms | bipolar; FSUSqrt(jk_trace=False, emit=False); cordiv-trace; identical encoded input stream; T=256, N=4000 |
| 2026-06-13 | operation.sqrt_tracejkff | FSUSqrt | yes (diff 0) | bit-exact CPU+MPS; spike_diff=0 over T=256, N=4000; napl & UnarySim RMSE-vs-sqrt identical (0.271) | 12.9 ms | 42.8 ms | bipolar; FSUSqrt(jk_trace=True, emit=False); identical encoded input stream; T=256, N=4000; decoded-value + per-timestep spike compare |
| 2026-06-13 | operation.sync_skewed | SkewedSync | yes (diff 0) | bit-exact, max\|diff\|=0 on CPU and MPS (per-timestep out_1 and out_2 spikes; 10000 elems x T=256) | 13.56 ms | 44.80 ms | unipolar; width=3 (cnt_max=7 = UnarySim depth=3); 10000 elements, T=256, sobol dims 1 & 3; in_1<in_2 enforced; counter sweeps min/max saturation |
| 2026-06-13 | operation.tanh_hard | FSUHardtanh | yes (diff 0, CPU and MPS) | bit-exact vs FSUHardtanh; RMSE 0.0000 vs analytic Hardtanh, SC bound 0.0625 | 4.8 ms | 11.8 ms | bipolar streaming, Identity pass-through; shape=(10000,) T=256; known-answer in-range inputs |
| 2026-06-13 | operation.tanh_hub | HUBHardtanh | yes (diff 0, CPU and MPS) | bit-exact vs HUBHardtanh (Hardtanh[-1,1]) including clip corners | 10.0 ms / 1000 calls (10007-elem) | 12.2 ms / 1000 calls (10007-elem) | single-shot binary-domain; inputs over [-3,3] exercising clip; corners {-2,-1,-0.5,0,0.3,1,2} |
| 2026-06-13 | operation.uni2bi | Uni2Bi | yes (diff 0) | bit-exact, 0 mismatched spikes over T=256 x N=10000, max\|diff\|=0.0, CPU+MPS | 4.92 ms | 20.18 ms | unipolar->bipolar stream reshaper; N=10000 T=256 width=3; deterministic (no RNG); napl accumulator never hits clamp bounds (0 clamp hits) so differing clamp constants vs UnarySim depth=8 are benign |

## 2026-07-24 comprehensive current-class CPU comparison

Fresh fixed-seed comparison against the current local UnarySim clone at
`/Users/diwu/Projects/UnarySim`. The run covers every public implementation class under
`sim/operation`, `sim/module`, and `sim/metric`: 40 operations, 26 modules, and 6 metrics.
The empty `operation.inhibit` and `module.wta` placeholders have no class to run.

- Correctness: 41 bit-exact, 18 within the stated stochastic bound, 4 differences, and
  9 current upstream references unavailable.
- Performance: NAPL was faster in 50 of 63 timed pairs and slower in 13. Speedup is
  `UnarySim time / NAPL time`, so values below 1 mean NAPL is slower.
- Workloads: streaming operations use 64 cycles and 4,096 elements; metrics use 64 cycles
  and 256 elements; streaming layers use 64 cycles on matched small tensors. Single-shot
  linear/RNN timings cover 200 calls and single-shot conv timings cover 20 calls. Each time
  is the median of five fresh-module runs on CPU.
- Inputs, weights, seeds, and spike streams are identical within each pair. The current
  UnarySim fixed-point/HUB/round code requires its historical float-shift behavior; the
  comparison supplies a temporary power-of-two shift compatibility shim without changing
  either repository.

| Class | Correctness | Max abs diff | NAPL CPU ms | UnarySim CPU ms | Speedup |
|-------|-------------|--------------|-------------|-----------------|---------|
| metric.accuracy | bit-exact | 0 | 0.172 | 0.171 | 0.99x |
| metric.correlation | bit-exact | 0 | 0.506 | 0.732 | 1.45x |
| metric.stability | bit-exact | 0 | 0.723 | 0.836 | 1.16x |
| metric.stability_builder | bit-exact | 0 | 0.577 | 1.355 | 2.35x |
| metric.stability_flux | bit-exact | 0 | 1.541 | 1.726 | 1.12x |
| metric.stability_norm | bit-exact | 0 | 0.932 | 1.163 | 1.25x |
| module.avgpool2d | bit-exact | 0 | 1.974 | 1.836 | 0.93x |
| module.conv | within bound (0.25) | 0.046875 | 7.287 | 9.825 | 1.35x |
| module.conv_fxp | DIFF | 0.78717 | 3.150 | 3.032 | 0.96x |
| module.conv_hub | DIFF | 0.0507812 | 3.926 | 4.081 | 1.04x |
| module.conv_pc | within bound (1.125) | 0.5 | 6.759 | 8.930 | 1.32x |
| module.conv_tlut | upstream unavailable | - | - | - | - |
| module.conv_ugemm | bit-exact | 0 | 2.480 | 9.192 | 3.71x |
| module.decoder | bit-exact | 0 | 0.370 | 0.169 | 0.46x |
| module.encoder | bit-exact | 0 | 0.189 | 0.319 | 1.69x |
| module.gru_hardnuapt | bit-exact | 0 | 3.809 | 5.015 | 1.32x |
| module.linear | within bound (0.25) | 0.046875 | 1.336 | 2.600 | 1.95x |
| module.linear_fxp | bit-exact | 0 | 6.208 | 4.171 | 0.67x |
| module.linear_gaines1 | within bound (0.25) | 0.0625 | 0.951 | 1.419 | 1.49x |
| module.linear_gaines2 | bit-exact | 0 | 0.798 | 1.220 | 1.53x |
| module.linear_gaines3 | within bound (0.25) | 0.109375 | 1.266 | 1.798 | 1.42x |
| module.linear_gaines4 | within bound (0.25) | 0.09375 | 0.696 | 1.547 | 2.22x |
| module.linear_hub | DIFF | 0.09375 | 8.117 | 9.492 | 1.17x |
| module.linear_pc | within bound (0.5) | 0.125 | 0.817 | 1.792 | 2.19x |
| module.linear_tlut | upstream unavailable | - | - | - | - |
| module.linear_ugemm | bit-exact | 0 | 1.811 | 2.126 | 1.17x |
| module.mgu | within bound (0.5) | 0.25 | 6.400 | 11.528 | 1.80x |
| module.mgu_hard | bit-exact | 0 | 3.036 | 6.593 | 2.17x |
| module.mgu_hardfxp | bit-exact | 0 | 20.964 | 31.605 | 1.51x |
| module.mgu_hardnua | bit-exact | 0 | 2.241 | 4.441 | 1.98x |
| module.mgu_hardpt | bit-exact | 0 | 3.441 | 9.734 | 2.83x |
| module.mgu_hub | within bound (0.5) | 0.375 | 7.363 | 13.030 | 1.77x |
| operation.add_any | within bound (0.125) | 0.015625 | 2.575 | 2.482 | 0.96x |
| operation.add_gaines | bit-exact | 0 | 0.171 | 0.419 | 2.45x |
| operation.add_ugemm | bit-exact | 0 | 2.246 | 2.423 | 1.08x |
| operation.bi2uni | DIFF | 0.140625 | 0.567 | 0.550 | 0.97x |
| operation.dff | bit-exact | 0 | 0.223 | 0.480 | 2.15x |
| operation.div_cordiv | within bound (0.375) | 0.109375 | 0.949 | 3.893 | 4.10x |
| operation.div_gaines | bit-exact | 0 | 0.810 | 2.226 | 2.75x |
| operation.div_iscb | within bound (0.5) | 0.203125 | 5.210 | 9.787 | 1.88x |
| operation.exp_n1 | bit-exact | 0 | 0.475 | 1.958 | 4.12x |
| operation.exp_ng | bit-exact | 0 | 0.498 | 0.952 | 1.91x |
| operation.gt_rc | upstream unavailable | - | - | - | - |
| operation.jkff | bit-exact | 0 | 0.551 | 1.079 | 1.96x |
| operation.lt_rc | upstream unavailable | - | - | - | - |
| operation.max_rc | upstream unavailable | - | - | - | - |
| operation.max_tc | upstream unavailable | - | - | - | - |
| operation.min_rc | upstream unavailable | - | - | - | - |
| operation.min_tc | upstream unavailable | - | - | - | - |
| operation.mul_and | bit-exact | 0 | 0.218 | 0.224 | 1.03x |
| operation.mul_csg | within bound (0.25) | 0.015625 | 3.858 | 4.469 | 1.16x |
| operation.mul_gaines | bit-exact | 0 | 0.218 | 0.226 | 1.03x |
| operation.relu_cnt | bit-exact | 0 | 0.541 | 0.692 | 1.28x |
| operation.relu_hub | bit-exact | 0 | 0.297 | 0.274 | 0.92x |
| operation.relu_sat | within bound (0.25) | 0.09375 | 1.339 | 0.686 | 0.51x |
| operation.round_fxp | bit-exact | 0 | 6.177 | 5.419 | 0.88x |
| operation.shiftreg | bit-exact | 0 | 0.137 | 0.558 | 4.08x |
| operation.sigmoid_hard | within bound (0.125) | 0.015625 | 0.692 | 0.899 | 1.30x |
| operation.sigmoid_hub | bit-exact | 0 | 0.541 | 1.127 | 2.08x |
| operation.signabs | bit-exact | 0 | 0.562 | 0.741 | 1.32x |
| operation.sqrt_emit | within bound (0.375) | 0.125 | 1.805 | 2.379 | 1.32x |
| operation.sqrt_gaines | bit-exact | 0 | 0.740 | 1.670 | 2.26x |
| operation.sqrt_traceiscb | within bound (0.375) | 0.1875 | 1.942 | 5.039 | 2.59x |
| operation.sqrt_tracejkff | within bound (0.375) | 0.171875 | 1.455 | 2.095 | 1.44x |
| operation.square_dff | upstream unavailable | - | - | - | - |
| operation.sync_skewed | bit-exact | 0 | 0.804 | 1.656 | 2.06x |
| operation.sync_skewed_int | bit-exact | 0 | 0.622 | 0.552 | 0.89x |
| operation.tanh_hard | bit-exact | 0 | 0.075 | 0.028 | 0.37x |
| operation.tanh_hub | bit-exact | 0 | 0.286 | 0.271 | 0.95x |
| operation.tanh_p1 | bit-exact | 0 | 1.474 | 2.047 | 1.39x |
| operation.tanh_pn | bit-exact | 0 | 0.518 | 0.945 | 1.82x |
| operation.uni2bi | bit-exact | 0 | 0.539 | 0.795 | 1.47x |

### Current differences

| Class | Evidence |
|-------|----------|
| `operation.bi2uni` | Decoded max absolute difference 0.140625. Current NAPL uses a bounded accumulator and `>= 1`; current UnarySim uses an unbounded accumulator plus a separate output accumulator and `>`. |
| `module.linear_hub` | Max absolute difference 0.09375. NAPL rounds scaled magnitudes before the HUB lookup; current UnarySim converts them to integer, which truncates. |
| `module.conv_hub` | Max absolute difference 0.0507812 from the same round-versus-truncate HUB level conversion. |
| `module.conv_fxp` | Max absolute difference 0.78717 on a valid input whose weight magnitude rounds below one. NAPL clamps with `2**width`; current UnarySim clamps with `2**(bitwidth-1)`, so this case reaches different quantized weights. |

Current upstream reference gaps: `min_rc`, `max_rc`, `lt_rc`, and `gt_rc` map most closely
to `FSUCompare`, whose current `forward` references an undefined `input`; UnarySim has no
standalone temporal comparator for `min_tc`/`max_tc`, no standalone square class for
`square_dff`, and no `TLUTLinear` or `TLUTConv2d` class in the current clone.

### NAPL slower pairs

| Class | Speedup |
|-------|---------|
| `operation.add_any` | 0.96x |
| `operation.bi2uni` | 0.97x |
| `operation.relu_sat` | 0.51x |
| `operation.relu_hub` | 0.92x |
| `operation.tanh_hub` | 0.95x |
| `operation.round_fxp` | 0.88x |
| `operation.sync_skewed_int` | 0.89x |
| `operation.tanh_hard` | 0.37x |
| `metric.accuracy` | 0.99x |
| `module.avgpool2d` | 0.93x |
| `module.decoder` | 0.46x |
| `module.linear_fxp` | 0.67x |
| `module.conv_fxp` | 0.96x |

Verification: `conda run -n napl python tests/sweep_test.py` completed with exit 0 and all
73 standalone test files passed. `git status --short -- src tests examples` remained empty
before and after the comparison.
