# uSystolic convnet_mnist (accuracy track) in napl

This example ports the **accuracy** track of UnarySim's `app/uSystolic/convnet_mnist`
into napl as a runnable example. uSystolic (HPCA 2022) measures how MNIST inference
accuracy degrades when the conv/linear layers of a trained FP32 CNN are replaced by
the uSystolic unary-MAC model (HUB) or a fixed-point model (FXP), swept over MAC
precision. Only accuracy is reproduced here; the cycle/energy/area numbers from the
paper live in a separate repo with external tooling that is not available.

Upstream source: `UnarySim/app/uSystolic/convnet_mnist/` (branch `stable-v0`).

## What it reuses

The HUB and FXP conv/linear cells are napl's existing single-shot, binary-domain
kernels, used as-is (nothing in `src/napl/` was modified):

- `napl.conv_hub`, `napl.linear_hub` -- the uSystolic unary-MAC model.
  Input and weight are quantized to sign-magnitude fixed point and each
  `|input| x |weight|` product is read from a precomputed unary-multiplication value
  map. Knob: `cycle` (the unary MAC cycle count).
- `napl.conv_fxp`, `napl.linear_fxp` -- the fixed-point model. Input and
  weight are dynamically scaled to `widthi`/`widthw`-bit fixed point, matmul'd, then
  shifted back. Knob: `bitwidth` (= widthi = widthw).

These layers process the whole tensor at once (no encoder/decoder streaming wiring),
so they compose with stock torch ops directly, exactly as UnarySim composes them.

## Files

- `mnist_data.py` -- MNIST loader with no torchvision dependency. Downloads and
  parses the raw IDX files from the same S3 mirror torchvision uses, with the
  standard MNIST normalization (mean 0.1307, std 0.3081).
- `model.py` -- the convnet in three forms: `ConvNetFP` (nn.Conv2d/nn.Linear),
  `ConvNetHUB` (conv_hub/linear_hub, `cycle` knob), `ConvNetFXP` (conv_fxp/linear_fxp,
  `bitwidth` knob). HUB/FXP load weights from the FP checkpoint; no retraining.
- `train_fp.py` -- trains `ConvNetFP` on MNIST (3 epochs, Adadelta lr=1.0, StepLR
  gamma=0.7) and saves `checkpoints/convnet_mnist.pt`.
- `eval_sweep.py` -- loads the checkpoint, reports FP32 accuracy, sweeps HUB over
  cycles and FXP over the matching bitwidths, and writes
  `results/convnet_mnist_result.csv`.

## Architecture (matches upstream exactly)

```text
conv1(1->32, 3x3)  relu
conv2(32->64, 3x3) relu
max_pool2d(2)
dropout(0.25)
flatten
fc1(9216->128)     relu
dropout(0.5)
fc2(128->10)
log_softmax
```

## Run (from the repo root, napl env)

```sh
conda run -n napl python zoo/usystolic/train_fp.py
conda run -n napl python zoo/usystolic/eval_sweep.py            # MPS if available, 2000 test imgs
conda run -n napl python zoo/usystolic/eval_sweep.py --test-size 10000 --device cpu  # full set on CPU
```

The scripts use sibling imports (`from model import ...`), so run them as shown
(the import resolves relative to the script directory).

## Cycle / bitwidth pairing

The upstream knob relationship is `cycle = 2**(bitwidth-1)`. The sweep uses the same
pairs: (cycle 32, bw 6), (64, 7), (128, 8), (256, 9), (512, 10), (1024, 11). `eval_sweep`
passes `width=bw` for each HUB point so the value map's `cycle_max = 2**(width-1)` equals
the requested cycle: a cycle above `2**(width-1)` is silently capped at that maximum, so
`cycle=1000` with `widthi=8` runs 128 cycles.

## Caveats

- **o-res / i-res:** upstream `FxpConv2d`/`FxpLinear` expose `keep_res` (input vs
  output) and `more_res` (input vs weight), so the committed CSV has both an
  `Fxp-o-res` and an `Fxp-i-res` column. napl's `conv_fxp`/`linear_fxp` have **no
  o-res/i-res split**: they always give input and weight the full `bitwidth`
  (`widthi == widthw == bitwidth`), which is exactly the upstream `keep_res="input"`
  (i-res) configuration. **This example reproduces only the i-res column.** The o-res
  configuration (output resolution, where the bitwidth is split between input and
  weight) is not reproducible without adding the split to the napl kernels.
- **No pooling op:** napl has no pooling primitive, so max-pooling stays stock
  `F.max_pool2d`. This is faithful to the upstream code, where pooling also sits
  unchanged between the HUB/FXP layers (those layers are single-shot binary-domain
  modules; pooling is a plain torch op in both repos).
- **Dropout at eval:** dropout is identity under `model.eval()`, so it is harmless in
  the HUB/FXP inference models and kept only for structural parity with upstream.

## Reproduced numbers vs upstream

FP32 baseline trained 3 epochs (98.93% on the full 10k test set). The sweep below was
run on MPS over the first 2000 test images (`--test-size 2000`), which finishes in
about a minute; use `--test-size 10000` for the full set.

napl (this example, 2000 test images, MPS):

| cycle | bitwidth | HUB top-1 | FXP top-1 |
|------:|---------:|----------:|----------:|
|    32 |        6 |    97.75% |    98.45% |
|    64 |        7 |    98.15% |    98.65% |
|   128 |        8 |    98.35% |    98.65% |
|   256 |        9 |    98.40% |    98.65% |
|   512 |       10 |    98.40% |    98.65% |
|  1024 |       11 |    98.40% |    98.70% |

FP32 baseline on the same 2000 images: 98.70%.

Upstream committed CSV (full 10k, longer FP training; `Fxp-i-res` is the comparable
FXP column):

| Cycle | uSys (HUB) | Fxp-i-res | Fxp-o-res |
|------:|-----------:|----------:|----------:|
|    32 |     98.94% |    99.02% |    95.46% |
|    64 |     98.98% |    99.02% |    98.04% |
|   128 |     99.03% |    99.04% |    98.86% |
|   256 |     99.07% |    99.04% |    98.98% |
|   512 |     99.06% |    99.04% |    98.87% |
|  1024 |     99.07% |    99.04% |    99.01% |
|    fp |     99.17% |    99.17% |    99.17% |

The reproduced trend matches the paper: HUB accuracy rises with the MAC cycle count
and approaches the FP baseline (97.75% at cycle 32 up to 98.40% at cycle 1024), and
FXP (i-res) stays within a small margin of FP at every bitwidth. The absolute numbers
sit a few tenths below upstream because this example trains only 3 epochs (vs the
upstream 14-epoch schedule) and evaluates a 2000-image subset by default; both gaps
close with `--epochs 14` and `--test-size 10000`.
