# NAPL Components

## operation

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| uni2bi | src/napl/operation/uni2bi.py | Convert unipolar spike trains to bipolar via scaled addition accumulator | no | no | UnaryAbs / Uni2Bi (bitstream conv helper) | yes | yes |
| min_rc | src/napl/operation/compare.py | Rate-coded min/argmin via sync_skewed + DFF | no | no | UnaryMin | no | yes |
| max_rc | src/napl/operation/compare.py | Rate-coded max/argmax via sync_skewed + DFF | no | no | UnaryMax | no | yes |
| lt_rc | src/napl/operation/compare.py | Rate-coded less-than comparison via sync_skewed + DFF | no | no | UnaryCompare (lt) | no | yes |
| gt_rc | src/napl/operation/compare.py | Rate-coded greater-than comparison via sync_skewed + DFF | no | no | UnaryCompare (gt) | no | yes |
| min_tc | src/napl/operation/compare.py | Temporal-coded min via AND gate | no | no | (temporal compare) | no | yes |
| max_tc | src/napl/operation/compare.py | Temporal-coded max via OR gate | no | no | (temporal compare) | no | yes |
| sigmoid_hard | src/napl/operation/sigmoid.py | FSU hard sigmoid (x+1)/2 via scaled add | no | no | UnarySigmoid / HardSigmoid | yes | yes |
| sigmoid_hub | src/napl/operation/sigmoid.py | Binary-domain hard sigmoid Hardsigmoid(x*scale) | no | no | HUBHardsigmoid | yes | skipped |
| square_dff | src/napl/operation/square.py | Unary square via AND/XNOR + DFF | no | no | UnarySquare | no | yes |
| shiftreg | src/napl/operation/shiftreg.py | Shift register (circular buffer FIFO) | no | no | ShiftReg | yes | yes |
| relu_cnt | src/napl/operation/relu.py | Bipolar rate-coded ReLU via counter vs bipolar 0 | no | no | UnaryReLU | yes | yes |
| relu_sat | src/napl/operation/relu.py | Bipolar rate-coded ReLU via saturating add | no | no | UnaryReLU | yes | yes |
| relu_hub | src/napl/operation/relu.py | Binary-domain ReLU as Hardtanh(0, scale) | no | no | HUBReLU | yes | skipped |
| _round_ste_fn | src/napl/operation/round.py | Straight-through fixed-point rounding autograd Function | no | yes | (Round STE helper) | no | no |
| round_fxp | src/napl/operation/round.py | Quantize to signed fixed-point via STE round, single-shot | no | no | num2tuple / Round fixed-point | yes | skipped |
| jkff | src/napl/operation/jkff.py | JK flip-flop | no | no | JKFF | yes | yes |
| mul_and | src/napl/operation/mul.py | Unary multiply via AND (unipolar) / XNOR (bipolar) | no | no | UnaryMul / GainesMul | yes | yes |
| mul_csg | src/napl/operation/mul.py | Unary multiply via conditional spike generation | no | no | UnaryMul (CSG) | yes | yes |
| dff | src/napl/operation/dff.py | D flip-flop (depth-configurable FIFO) | no | no | DFF | yes | yes |
| div_cordiv | src/napl/operation/div.py | Correlated division (unipolar), CORDIV kernel | no | no | CORDIV / UnaryDiv | yes | yes |
| div_iscb | src/napl/operation/div.py | In-stream correlation-based division (rate coded) | no | no | UnaryDiv (ISCBDIV) | yes | yes |
| sqrt_tracejkff | src/napl/operation/sqrt.py | Square root via stochastic bit-inserting using JKFF | no | no | UnarySqrt | yes | yes |
| sqrt_traceiscb | src/napl/operation/sqrt.py | Square root via bit-inserting using ISCBDIV | no | no | UnarySqrt | yes | yes |
| sqrt_emit | src/napl/operation/sqrt.py | Square root via opportunistic bit-inserting (best accuracy) | no | no | UnarySqrt | yes | yes |
| sign_abs | src/napl/operation/sign_abs.py | Sign + magnitude of bipolar rate-coded spikes | no | no | UnaryAbs / SignMag | yes | yes |
| add_any | src/napl/operation/add.py | Any-scale unary addition with saturating accumulator | no | no | UnaryAdd / GainesAdd | yes | yes |
| bi2uni | src/napl/operation/bi2uni.py | Convert bipolar spike trains to unipolar via non-scaled add | no | no | Bi2Uni | yes | yes |
| tanh_hard | src/napl/operation/tanh.py | FSU hard tanh (identity pass-through on spikes) | no | no | UnaryTanh | yes | yes |
| tanh_hub | src/napl/operation/tanh.py | Binary-domain hard tanh Hardtanh(-1,1) | no | no | HUBHardtanh | yes | skipped |
| sync_skewed | src/napl/operation/sync.py | Skewed synchronizer of two spike trains | no | no | Skewed Sync / Synchronizer | yes | yes |

## module

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| mgu_hard | src/napl/module/rnn.py | Binary-domain Minimal Gated Unit cell with hard activations | no | no | HardMGUCell | yes | n/a (operation-only) |
| mgu_hardfxp | src/napl/module/rnn.py | MGU cell with fixed-point quantized operands (STE) | no | no | HardMGUCellFXP | yes | n/a (operation-only) |
| mgu_fsu | src/napl/module/rnn.py | Streaming (FSU) MGU cell, one timestep at a time | no | no | FSUMGUCell | yes | n/a (operation-only) |
| mgu_hub | src/napl/module/rnn.py | Hybrid MGU: encode, run mgu_fsu, decode | no | no | HUBMGUCell | yes | n/a (operation-only) |
| conv_fxp | src/napl/module/conv.py | Binary-domain fixed-point conv2d (im2col + linear_fxp) | no | no | FXPConv2d | yes | n/a (operation-only) |
| conv_hub | src/napl/module/conv.py | Binary-domain HUB conv2d (value-map kernel) | no | no | HUBConv2d | yes | n/a (operation-only) |
| conv_tlut | src/napl/module/conv.py | Binary-domain temporal-LUT conv2d | no | no | TLUTConv2d | yes | n/a (operation-only) |
| conv_fsu | src/napl/module/conv.py | Streaming (FSU) unary conv2d | no | no | FSUConv2d | yes | n/a (operation-only) |
| conv_fsu_pc | src/napl/module/conv_fsu_pc.py | Streaming (FSU) unary conv2d parallel counter | no | no | FSUConv2dPC | yes | n/a (operation-only) |
| encoder | src/napl/module/encoder.py | Number to spike stream via RNG comparison | no | no | RateCoding / BSGen / SourceGen | yes | n/a (operation-only) |
| linear_fsu | src/napl/module/linear.py | Streaming (FSU) unary fully-connected layer | no | no | FSULinear | yes | n/a (operation-only) |
| linear_fsu_pc | src/napl/module/linear.py | Streaming (FSU) unary linear parallel counter | no | no | FSULinearPC | yes | n/a (operation-only) |
| _linear_fxp_fn | src/napl/module/linear.py | Fixed-point linear forward + STE backward autograd Function | no | yes | (FXPLinear STE) | no | n/a (operation-only) |
| linear_fxp | src/napl/module/linear.py | Binary-domain fixed-point fully-connected layer | no | no | FXPLinear | yes | n/a (operation-only) |
| _linear_hub_fn | src/napl/module/linear.py | HUB linear forward + STE backward autograd Function | no | yes | (HUBLinear STE) | no | n/a (operation-only) |
| linear_hub | src/napl/module/linear.py | Binary-domain HUB fully-connected layer | no | no | HUBLinear | yes | n/a (operation-only) |
| _linear_tlut_fxpfxp_fn | src/napl/module/linear.py | TLUT fxp/fxp linear autograd Function (STE) | no | yes | (TLUTLinear STE) | no | n/a (operation-only) |
| _linear_tlut_fxpfp_fn | src/napl/module/linear.py | TLUT fxp/fp linear autograd Function (STE) | no | yes | (TLUTLinear STE) | no | n/a (operation-only) |
| _linear_tlut_fpfp_fn | src/napl/module/linear.py | TLUT fp/fp linear autograd Function (STE) | no | yes | (TLUTLinear STE) | no | n/a (operation-only) |
| linear_tlut | src/napl/module/linear.py | Binary-domain temporal-LUT fully-connected layer | no | no | TLUTLinear | yes | n/a (operation-only) |
| decoder | src/napl/module/decoder.py | Spike stream to number by counting | no | no | (decoder / ProgError counterpart) | yes | n/a (operation-only) |

## metric

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| correlation | src/napl/metric/correlation.py | Stochastic cross-correlation (SCC) over spike streams | no | no | Correlation / ProgressiveError SCC | yes | n/a (operation-only) |
| accuracy | src/napl/metric/accuracy.py | Progressive precision / progressive error metric | no | no | ProgError / ProgressivePrecision | yes | n/a (operation-only) |
| stability | src/napl/metric/stability.py | Per-element normalized stability of a spike stream | no | no | Stability / NSC | yes | n/a (operation-only) |

## structure

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|

No class definitions found; all structure modules (axon, soma, dendrite, synapse, receptor, column) are empty placeholders.

## algorithm

| Class | File (repo-relative) | Description | Placeholder? | Autograd helper? | UnarySim counterpart | Validated? | RTL? |
|---|---|---|---|---|---|---|---|
| butterfly_spike | src/napl/algorithm/fft/butterfly.py | Spike-domain FFT butterfly (encoder + mul_csg + add + decoder) | no | no | (FFT butterfly, no direct counterpart) | no | n/a (operation-only) |
| butterfly_binary | src/napl/algorithm/fft/butterfly.py | Binary reference FFT butterfly (plain torch.nn.Module) | no | no | (FFT butterfly reference) | no | n/a (operation-only) |

---

Total: 57 classes across 5 subpackages (31 operation, 21 module, 3 metric, 0 structure, 2 algorithm). Of these, 7 are torch.autograd.Function STE/backward helpers (_round_ste_fn, _linear_fxp_fn, _linear_hub_fn, _linear_tlut_fxpfxp_fn, _linear_tlut_fxpfp_fn, _linear_tlut_fpfp_fn). 42 classes are validated against UnarySim, and 26 operation classes have a verified RTL row.
