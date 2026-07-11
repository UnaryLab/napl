# uBrain in napl: Cascade CNN+RNN, FP and HUB (unary) forms

This example ports the **uBrain** application from UnarySim
(`app/uBrain`, the `Cascade_CNN_RNN` model and its functional unary path) into
napl as a runnable example, and reproduces what is reproducible **without** the
proprietary EEG datasets or the EDA/hardware tooling.

## What uBrain is

uBrain is a brain-computer-interface (BCI) accelerator that classifies EEG with
dynamic rate-coded unary computing. The network is a cascade CNN + RNN:

```
conv1 (1 -> 16ch, 3x3, pad 1) -> ScaleReLU
conv2 (16 -> 32ch, 3x3, pad 1) -> ScaleReLU
flatten -> fc3 (-> 256) -> ScaleReLU -> dropout
reshape to (win=10, fc_sz=256) -> MGU (minimal gated unit) cell looped 10 steps
fc5 (hidden 64 -> sum(num_class)) -> Hardtanh
-> two heads: [5, 2]  (MI 5-class + SP 2-class)
```

Defaults here follow the **10-10 MI** configuration: `input_sz=[10, 11]`,
`rnn_win_sz=10`, `fc_sz=256`, `rnn_hidden_sz=64`, `num_class=[5, 2]`, `bias=False`.
With same-padding conv (`padding=1`, no pooling), fc3 sees
`10 * 11 * 32 = 3520` input features. EEG input is shaped `(batch, win, h, w)`.

Upstream has four variants: `fp` (trainable float), `fxp` (round-quantized),
`hub` (hybrid unary-binary, weights from the FP checkpoint), and `hub_noisy`.
Training is FP-only; fxp/hub are inference-only and reuse the FP weights.

## What was ported (this directory)

- **`model.py`** - two faithful forms of `Cascade_CNN_RNN`:
  - `Cascade_CNN_RNN_FP`: `nn.Conv2d` / `nn.Linear` + `mgu_hard` +
    `relu_hub`/`tanh_hub` (float domain, trainable) == UnarySim `model_fp`.
  - `Cascade_CNN_RNN_HUB`: `conv_hub` / `linear_hub` + `mgu_hub` +
    `relu_hub`/`tanh_hub` (hybrid unary-binary inference) == UnarySim `model_hub`.
  - `build_hub_from_fp(fp_model, width, rng)`: builds a HUB model that **shares
    the FP model's weights** (conv/linear `nn.weight` -> HUB `weight_ext`;
    `mgu_hard.weight_f/weight_n` -> `mgu_hub.weight_f/weight_n`), so the two are
    directly comparable.
- **`run_fp.py`** - FP forward on EEG-shaped random input on CPU and MPS, plus a
  clearly-labeled synthetic-data train smoke loop (random labels).
- **`eval_hub_fidelity.py`** - FP-vs-HUB error of the network output, swept over
  the unary bitwidth (cycles). Writes `results/hub_fidelity.csv`.

### Reused from napl (NOT re-ported)

| UnarySim            | napl                         |
|---------------------|------------------------------|
| `nn.Conv2d`/`Linear`| `conv_hub` / `linear_hub` (HUB path) |
| `HardMGUCell`       | `mgu_hard` (FP) / `mgu_hub` (HUB) |
| `ScaleReLU`         | `relu_hub`                   |
| `nn.Hardtanh`       | `tanh_hub`                   |
| `truncated_normal`  | `napl.utils.truncated_normal`|
| `ProgError`/RMSE    | `napl.metric.report_error`   |

`mgu_hub` internally encodes its inputs, streams the `mgu_fsu` cell over
`2**width` cycles, and decodes via the progressive-error metric.

## What IS and IS NOT reproducible here

**NOT reproducible (and not attempted, no numbers fabricated):**
- **EEG classification accuracy.** The PhysioNet (MI) and neonatal (SP) EEG
  datasets are not present and not downloadable here, and no trained checkpoint
  is committed. All inputs and labels in these scripts are **random**, so nothing
  here is a classification result.
- **Hardware results (area / power / energy).** These need Synopsys DC + a 32nm
  PDK, a Jetson Nano, and external SCALE-Sim / uSystolic-Sim, none available.

**Reproducible (what these scripts actually show):**
1. The `Cascade_CNN_RNN` model runs **end-to-end** in napl in both the **FP** and
   the **HUB (unary)** form on EEG-shaped input, with the correct per-head output
   shapes.
2. A **fidelity** check: the HUB (unary) output tracks the FP output, and the
   error **shrinks as the bitwidth/cycle count grows** (the spirit of upstream
   `layer_eval/` RMSE-vs-bitwidth and `model_hub`'s ProgError reporting), applied
   to the whole-network output.

## Run commands (in order)

> **Env note.** If torch emits a duplicate-OpenMP warning, set
> `KMP_DUPLICATE_LIB_OK=TRUE` (benign on macOS).

```sh
cd examples/ubrain
conda run -n napl python run_fp.py
conda run -n napl python eval_hub_fidelity.py
```

## Real numbers obtained

EEG input shape `(batch, win=10, h=10, w=11)`. FP uses `batch=4`; the HUB sweep
uses `batch=2` (the HUB MGU streams `2**width` cycles per step, looped over 10
steps, so it is the cost driver).

**FP forward (`run_fp.py`):** runs on both devices. Flat output `(batch, 7)`,
heads `(batch, 5)` and `(batch, 2)`. Forward ~7.5 ms on CPU, ~616 ms on MPS
(first-call MPS warmup). The 3-step synthetic train loop (random labels) executes
and updates weights (loss ~2.31, **not** a result).

**FP-vs-HUB fidelity (`eval_hub_fidelity.py`, CPU, batch=2, rng=sobol):**

| width | cycles | rmse    | max_abs_err | seconds |
|------:|-------:|--------:|------------:|--------:|
| 6     | 64     | 0.24256 | 0.41572     | 0.12    |
| 8     | 256    | 0.03656 | 0.07204     | 0.38    |
| 10    | 1024   | 0.01973 | 0.03881     | 1.44    |

RMSE falls monotonically as cycles grow (error roughly halves per +2 bits),
consistent with the unary `~1/sqrt(N)` bound. Saved to `results/hub_fidelity.csv`.

**Per-device result:**
- **CPU:** FP and HUB both run, all widths.
- **MPS:** the **FP** model and `conv_hub`/`linear_hub`/`relu_hub` run fine. The
  **HUB MGU does not run on MPS** (see kernel gap below); the fidelity sweep is a
  CPU result, with the MPS HUB path sanity-checked and reported as a known gap.

## napl kernel issue found

**`mgu_hub` does not support non-CPU devices.** Its `forward()` constructs its
internal submodules (`encoder`, `mgu_fsu`, `accuracy`) on the default (CPU) device
and never moves them to the input tensor's device, so the inner `add_any`
accumulator stays on CPU while the inputs are on MPS:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, mps:0 and cpu!
```

This is reproducible in isolation (a bare `mgu_hub(...).to('mps')` call on an MPS
input fails the same way). `conv_hub` / `linear_hub` / `relu_hub` / `tanh_hub`
and the whole FP model run correctly on MPS, so the gap is specific to `mgu_hub`
(and would also affect CUDA). A fix would build the inner modules on
`input.device` (or `.to(input.device)` them) inside `mgu_hub.forward`. Per the
task rules, `src/napl/` was not modified; this is reported only.

No issues found with `conv_hub`, `linear_hub`, `relu_hub`, `tanh_hub`, or the FP
`mgu_hard`. No pooling/neuron/WTA primitive was needed: uBrain's conv path is
same-padding stride-1 with no pooling.
