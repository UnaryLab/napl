# NAPL Components

Inventory of every class in `src/napl/{operation,module,metric,structure,algorithm}`. Validated? is derived from `reports/napl-validate-unarysim-report.md`; RTL? from `reports/napl-gen-rtl-report.md` (via `ledger_status.py`). UnarySim counterparts checked against the local clone at `/Users/diwu/Projects/UnarySim`.

## operation

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| add_any | src/napl/sim/operation/add_any.py | Any-scale streaming unary addition (unipolar/bipolar) | no | no | FSUAdd | yes | yes |
| add_gaines | src/napl/sim/operation/add_gaines.py | Gaines addition: MUX-select scaled sum or OR-gate non-scaled sum | no | no | GainesAdd | no | no |
| add_ugemm | src/napl/sim/operation/add_ugemm.py | uGEMM-style scaled/non-scaled addition via running spike-count carry emission | no | no | FSUAdduGEMM | no | no |
| bi2uni | src/napl/sim/operation/bi2uni.py | Bipolar-to-unipolar stream conversion via non-scaled addition | no | no | Bi2Uni | yes | yes |
| min_rc | src/napl/sim/operation/min_rc.py | Min and argmin of rate-coded streams using sync_skewed | no | no | FSUCompare | no | yes |
| max_rc | src/napl/sim/operation/max_rc.py | Max and argmax of rate-coded streams using sync_skewed | no | no | FSUCompare | no | yes |
| lt_rc | src/napl/sim/operation/lt_rc.py | Less-than comparison of rate-coded streams using sync_skewed | no | no | FSUCompare | no | yes |
| gt_rc | src/napl/sim/operation/gt_rc.py | Greater-than comparison of rate-coded streams using sync_skewed | no | no | FSUCompare | no | yes |
| min_tc | src/napl/sim/operation/min_tc.py | Min of temporal-coded streams via AND gate | no | no |  | no | yes |
| max_tc | src/napl/sim/operation/max_tc.py | Max of temporal-coded streams via OR gate | no | no |  | no | yes |
| dff | src/napl/sim/operation/dff.py | D flip-flop (one-timestep spike delay) | no | no |  | yes | yes |
| div_cordiv | src/napl/sim/operation/div_cordiv.py | Correlated division (CORDIV), unipolar, synchronized operands | no | no | CORDIV_kernel | yes | yes |
| div_iscb | src/napl/sim/operation/div_iscb.py | In-stream correlation-based division (rate coding) | no | no | FSUDiv | yes | yes |
| div_gaines | src/napl/sim/operation/div_gaines.py | Gaines division: saturating up/down counter drives quotient vs RNG | no | no | GainesDiv | no | no |
| exp_n1 | src/napl/sim/operation/exp_n1.py | Unary exp(-x) via truncated Maclaurin series NAND/AND chain with delay taps | no | no | expN1 | no | no |
| exp_ng | src/napl/sim/operation/exp_ng.py | FSM-based exp(-2*gain*x) via saturating up/down counter (Brown-Card) | no | no | expNG | no | no |
| jkff | src/napl/sim/operation/jkff.py | JK flip-flop | no | no | JKFF | yes | yes |
| mul_and | src/napl/sim/operation/mul_and.py | Unary multiplication: AND (unipolar) / XNOR (bipolar) | no | no | FSUMul | yes | yes |
| mul_csg | src/napl/sim/operation/mul_csg.py | Unary multiplication via conditional spike generation (uGEMM CSG) | no | no | FSUMul | yes | yes |
| mul_gaines | src/napl/sim/operation/mul_gaines.py | Gaines stochastic multiplication, gate-identical to mul_and | no | no | GainesMul | no | no |
| relu_cnt | src/napl/sim/operation/relu_cnt.py | ReLU by comparing counter spike value against bipolar zero | no | no | FSUReLU | yes | yes |
| relu_sat | src/napl/sim/operation/relu_sat.py | ReLU by saturating the spike value to 0 | no | no | FSUReLU | yes | yes |
| relu_hub | src/napl/sim/operation/relu_hub.py | Binary-domain ReLU clipping to [0, scale], single-shot | no | no | ScaleReLU | yes | skipped |
| _round_ste_fn | src/napl/sim/operation/round_fxp.py | Straight-through fixed-point round/floor/ceil autograd helper | no | yes | RoundingNoGrad | no | no |
| round_fxp | src/napl/sim/operation/round_fxp.py | Signed fixed-point quantizer with straight-through rounding, single-shot | no | no | Round | yes | skipped |
| shiftreg | src/napl/sim/operation/shiftreg.py | Shift register over spike streams | no | no | ShiftReg | yes | yes |
| sigmoid_hard | src/napl/sim/operation/sigmoid_hard.py | Streaming hard sigmoid as scaled addition (x+1)/2 | no | no | FSUHardsigmoid | yes | yes |
| sigmoid_hub | src/napl/sim/operation/sigmoid_hub.py | Binary-domain hard sigmoid Hardsigmoid(x*scale), single-shot | no | no | ScaleHardsigmoid | yes | skipped |
| signabs | src/napl/sim/operation/signabs.py | Sign and magnitude of bipolar rate-coded spikes | no | no | FSUAbs | yes | yes |
| sqrt_tracejkff | src/napl/sim/operation/sqrt_tracejkff.py | Square root via stochastic bit inserting using a JKFF trace | no | no | FSUSqrt | yes | yes |
| sqrt_traceiscb | src/napl/sim/operation/sqrt_traceiscb.py | Square root via stochastic bit inserting using iscb division trace | no | no | FSUSqrt | yes | yes |
| sqrt_emit | src/napl/sim/operation/sqrt_emit.py | Square root via opportunistic bit inserting (best accuracy of the three) | no | no | FSUSqrt | yes | yes |
| sqrt_gaines | src/napl/sim/operation/sqrt_gaines.py | Gaines square root via saturating up/down counter with squared-output feedback | no | no | GainesSqrt | no | no |
| square_dff | src/napl/sim/operation/square_dff.py | Unary square with AND gate plus DFF-delayed input | no | no | FSUMul | no | yes |
| sync_skewed | src/napl/sim/operation/sync_skewed.py | Skewed synchronizer of two spike streams | no | no | SkewedSync | yes | yes |
| sync_skewed_int | src/napl/sim/operation/sync_skewed_int.py | Skewed synchronizer using integer stochastic computing | no | no | SkewedSyncInt | no | no |
| tanh_hard | src/napl/sim/operation/tanh_hard.py | Streaming hard tanh via saturating scaled addition | no | no | FSUHardtanh | yes | yes |
| tanh_hub | src/napl/sim/operation/tanh_hub.py | Binary-domain hard tanh clipping to [-1, 1], single-shot | no | no | ScaleHardtanh | yes | skipped |
| tanh_p1 | src/napl/sim/operation/tanh_p1.py | Combinational tanh via Parhi series-expansion NAND/AND cascade | no | no | tanhP1 | no | no |
| tanh_pn | src/napl/sim/operation/tanh_pn.py | FSM-based tanh(N*x/2) with 2**depth states (Brown-Card) | no | no | tanhPN | no | no |
| uni2bi | src/napl/sim/operation/uni2bi.py | Unipolar-to-bipolar stream conversion via scaled addition | no | no | Uni2Bi | yes | yes |

## module

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| avgpool2d | src/napl/sim/module/avgpool2d.py | Streaming unary 2d average pooling via scaled addition | no | no | FSUAvgPool2d | no | n/a (operation-only) |
| conv_fxp | src/napl/sim/module/conv_fxp.py | Binary-domain fixed-point conv2d (im2col + linear_fxp + fold), STE-trainable | no | no | FxpConv2d | yes | n/a (operation-only) |
| conv_hub | src/napl/sim/module/conv_hub.py | Binary-domain HUB conv2d with unary value-map multiplication, STE-trainable | no | no | HUBConv2d | yes | n/a (operation-only) |
| conv_tlut | src/napl/sim/module/conv_tlut.py | Binary-domain temporal-LUT conv2d (fxpfxp/fxpfp/fpfp), STE-trainable | no | no |  | yes | n/a (operation-only) |
| conv | src/napl/sim/module/conv.py | Streaming unary conv2d, weights encoded on a distinct RNG dim | no | no | FSUConv2d | yes | n/a (operation-only) |
| conv_pc | src/napl/sim/module/conv_pc.py | Streaming conv2d parallel counter: per-timestep binary inner-product counts | no | no | FSUConv2dPC | yes | n/a (operation-only) |
| conv_ugemm | src/napl/sim/module/conv_ugemm.py | Streaming conv2d with uGEMM-style conditional spike generation | no | no | FSUConv2duGEMM | no | n/a (operation-only) |
| decoder | src/napl/sim/module/decoder.py | Spike-stream-to-number decoder by counting spikes | no | no | ProgError | yes | n/a (operation-only) |
| encoder | src/napl/sim/module/encoder.py | Number-to-spike-stream encoder via comparison against an RNG num_seq | no | no | BSGen | yes | n/a (operation-only) |
| gru_hardnuapt | src/napl/sim/module/gru_hardnuapt.py | Binary-domain GRU cell with hard activations, non-unary-aware, PyTorch layout | no | no | HardGRUCellNUAPT | no | n/a (operation-only) |
| linear | src/napl/sim/module/linear.py | Streaming unary fully-connected layer, bit-by-bit W x (+ b) | no | no | FSULinear | yes | n/a (operation-only) |
| linear_pc | src/napl/sim/module/linear_pc.py | Streaming linear parallel counter: per-timestep binary inner-product counts | no | no | FSULinearPC | yes | n/a (operation-only) |
| _linear_fxp_fn | src/napl/sim/module/_shared.py | STE autograd helper for linear_fxp | no | yes | FxpLinearFunction | no | n/a (operation-only) |
| linear_fxp | src/napl/sim/module/linear_fxp.py | Binary-domain fixed-point fully-connected layer, STE-trainable | no | no | FxpLinear | yes | n/a (operation-only) |
| _linear_hub_fn | src/napl/sim/module/_shared.py | STE autograd helper for linear_hub | no | yes | HUBLinearFunction | no | n/a (operation-only) |
| linear_hub | src/napl/sim/module/linear_hub.py | Binary-domain HUB linear via precomputed unary-product value map, STE-trainable | no | no | HUBLinear | yes | n/a (operation-only) |
| _linear_tlut_fxpfxp_fn | src/napl/sim/module/_shared.py | STE autograd helper for linear_tlut fxpfxp mode | no | yes |  | no | n/a (operation-only) |
| _linear_tlut_fxpfp_fn | src/napl/sim/module/_shared.py | STE autograd helper for linear_tlut fxpfp mode | no | yes |  | no | n/a (operation-only) |
| _linear_tlut_fpfp_fn | src/napl/sim/module/_shared.py | STE autograd helper for linear_tlut fpfp mode | no | yes |  | no | n/a (operation-only) |
| linear_tlut | src/napl/sim/module/linear_tlut.py | Binary-domain temporal-LUT linear (fxpfxp/fxpfp/fpfp), STE-trainable | no | no |  | yes | n/a (operation-only) |
| linear_ugemm | src/napl/sim/module/linear_ugemm.py | Streaming linear with uGEMM-style conditional spike generation from binary weights | no | no | FSULinearuGEMM | no | n/a (operation-only) |
| linear_gaines1 | src/napl/sim/module/linear_gaines1.py | Streaming Gaines linear, gMUL + gADD | no | no | GainesLinear1 | no | n/a (operation-only) |
| linear_gaines2 | src/napl/sim/module/linear_gaines2.py | Streaming Gaines linear, gMUL + uADD (per-column Sobol dims) | no | no | GainesLinear2 | no | n/a (operation-only) |
| linear_gaines3 | src/napl/sim/module/linear_gaines3.py | Streaming Gaines linear, uMUL (CSG) + gADD | no | no | GainesLinear3 | no | n/a (operation-only) |
| linear_gaines4 | src/napl/sim/module/linear_gaines4.py | Streaming Gaines linear, gMUL + gADD, LFSR-flavored decorrelation | no | no | GainesLinear4 | no | n/a (operation-only) |
| mgu_hardnua | src/napl/sim/module/mgu_hardnua.py | Non-unary-aware hard MGU cell (no hard-tanh range clamps), single-shot | no | no | HardMGUCellNUA | no | n/a (operation-only) |
| mgu_hardpt | src/napl/sim/module/mgu_hardpt.py | Hard MGU cell in PyTorch RNNCell style with unary-range clamps, single-shot | no | no | HardMGUCellPT | no | n/a (operation-only) |
| mgu_hard | src/napl/sim/module/mgu_hard.py | Binary-domain MGU cell with hard activations, single-shot, trainable | no | no | HardMGUCell | yes | n/a (operation-only) |
| mgu_hardfxp | src/napl/sim/module/mgu_hardfxp.py | mgu_hard with fixed-point rounding on every operand (quant-aware STE) | no | no | HardMGUCellFxp | yes | n/a (operation-only) |
| mgu | src/napl/sim/module/mgu.py | Streaming MGU cell composed of streaming primitives, one timestep at a time | no | no | FSUMGUCell | yes | n/a (operation-only) |
| mgu_hub | src/napl/sim/module/mgu_hub.py | Hybrid MGU: encodes operands, runs mgu over 2**width cycles, decodes | no | no | HUBMGUCell | yes | n/a (operation-only) |

## metric

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| accuracy | src/napl/sim/metric/accuracy.py | Progressive accuracy/error of a spike stream (progressive precision) | no | no | ProgError | yes | n/a (operation-only) |
| correlation | src/napl/sim/metric/correlation.py | Stochastic cross-correlation (SCC) between spike streams | no | no | Correlation | yes | n/a (operation-only) |
| stability | src/napl/sim/metric/stability.py | Per-element stability: fraction of run after error last exceeded threshold | no | no | Stability | yes | n/a (operation-only) |
| stability_builder | src/napl/sim/metric/stability_builder.py | Generates a spike stream with a prescribed normalized stability | no | no | NSbuilder | no | n/a (operation-only) |
| stability_norm | src/napl/sim/metric/stability_norm.py | Normalized, value-independent stability of a spike stream | no | no | NormStability | no | n/a (operation-only) |

## structure

No classes. All files (`axon.py`, `column.py`, `dendrite.py`, `receptor.py`, `soma.py`, `synapse.py`) are empty placeholders, and `structure/__init__.py` imports a nonexistent `minicolumn`, so `import napl.sim.structure` currently fails (known, per CLAUDE.md).

## algorithm

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| butterfly_spike | src/napl/sim/algorithm/fft/butterfly_spike.py | FFT butterfly in the spike domain (mul_csg + add_any over encoder/decoder) | no | no |  | no | n/a (operation-only) |
| butterfly_binary | src/napl/sim/algorithm/fft/butterfly_binary.py | Binary-domain reference FFT butterfly (plain torch.nn.Module) | no | no |  | no | n/a (operation-only) |

`algorithm/fft/fft.py` and `module/wta.py` are empty placeholders (no classes).

## Totals

79 classes total (42 operation, 31 module, 5 metric, 0 structure, 2 algorithm); 6 torch.autograd.Function STE helpers; 42 already validated against UnarySim; 26 operation classes with verified RTL (plus 4 recorded as skipped: relu_hub, round_fxp, sigmoid_hub, tanh_hub).
