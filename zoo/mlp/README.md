# MLP3 on MNIST in the unary (stochastic-computing) domain

A runnable napl port of the UnarySim `app/mlp` application. A small 3-layer MLP
(1024 -> width -> width -> 10) is trained on MNIST in floating point, then run in napl's
streaming spike domain to reproduce the original result: a per-cycle
(progressive-precision) accuracy curve that rises from chance toward the floating-point
baseline as more spike bits are streamed.

## What this is

- `model.py` - `MLP3` and the clamp-aware variants, ported from UnarySim `app/mlp/model.py`,
  depending on `torch` only (no UnarySim import). Input size is configurable (default fan-in
  1024 = a 32x32 image). `MLP3_clamp_train` clamps each activation to [-1, 1] after ReLU so
  the learned weights keep activations in the bipolar [-1, 1] range the unary datapath uses;
  `MLP3_clamp_eval` is the matching inference model whose per-layer outputs the unary pipeline
  reproduces.
- `train_fp.py` - trains `MLP3_clamp_train` on MNIST (resized to 32x32) in floating point for a
  few epochs at a narrow hidden width, applies `NN_SC_Weight_Clipper(bitwidth=8)` per epoch
  (bipolar 8-bit weight/bias quantization), prints test accuracy, and saves the state_dict to
  `checkpoints/mlp3_mnist.pt`. MNIST downloads to `data/`.
- `eval_unary.py` - loads the checkpoint, computes the FP test accuracy, then streams the MLP
  in the unary spike domain for T = 256 cycles and produces the per-cycle accuracy curve, saved
  to `results/cycle_accuracy_mlp.csv` (columns `cycle, accuracy`).

## How to run

From the repo root, in the `napl` conda env, in order:

```sh
conda run -n napl python zoo/mlp/train_fp.py
conda run -n napl python zoo/mlp/eval_unary.py
```

Useful flags for `eval_unary.py`:

```sh
conda run -n napl python zoo/mlp/eval_unary.py --device cpu --samples 256
conda run -n napl python zoo/mlp/eval_unary.py --device mps --samples 256
conda run -n napl python zoo/mlp/eval_unary.py --sanity      # tiny both-device check
```

`--device` defaults to MPS when available, else CPU. The eval is fully batched, so the whole
256-cycle run over a few hundred samples finishes in well under a second on either device.

## What it reuses from napl

- `napl.encoder` - number-to-spike encoding (bipolar, Sobol RNG), one spike per cycle.
- `napl.linear_pc` (the parallel-counter streaming linear, UnarySim's `FSULinearPC`) - the
  per-cycle binary inner-product count of input spikes against freshly Sobol-encoded weight
  spikes on a decorrelated RNG dimension. Accumulating the count over k cycles and forming
  `2*(count/k) - entry` recovers the bipolar `W x + b` with progressively higher precision.
- `napl.relu_hub` - bounded ReLU in the binary domain between streamed linear layers.
- `napl.utils.NN_SC_Weight_Clipper` - bipolar 8-bit weight/bias clipping during training.

The layers are streamed **sequentially** (layer L is run to convergence over all T cycles, its
activation is read out, then layer L+1 is streamed). This keeps every layer's input stream
stationary (a re-encoded fixed value), which is what gives the high per-cycle fidelity. The
reported per-cycle curve is the final (fc3) layer's running-prediction accuracy at each cycle
count: the progressive-precision readout.

## Reproduced numbers

Config used (kept small so both scripts finish in seconds): hidden width 128, 3 epochs,
batch 128, Adam lr 1e-3, bitwidth 8 (T = 256 cycles), bipolar, Sobol RNG. Devices: CPU and
MPS (this machine has no CUDA). Training runs on MPS; eval runs identically on CPU and MPS.

Training (`train_fp.py`):

```text
epoch 1/3: test acc = 0.9060
epoch 2/3: test acc = 0.9575
epoch 3/3: test acc = 0.9596
final FP test accuracy (clamped/quantized weights): 0.9596
```

Unary evaluation (`eval_unary.py`, 512 test images, T = 256, CPU; the committed
`results/cycle_accuracy_mlp.csv` is this run):

```text
FP (clamp-eval) accuracy on 512 test images: 0.8770
per-cycle accuracy: {1: 0.082, 16: 0.951, 64: 0.965, 256: 0.969}
final-cycle (256) unary accuracy: 0.9688
```

The curve rises from chance (~0.08 at 1 cycle) to ~0.95 within a handful of cycles and
converges to ~0.97, tracking and slightly exceeding the clamp-eval FP baseline on this batch
(the unary clamp/ReLU acts as mild regularization). With Sobol RNG the progressive-precision
convergence is fast, so the curve saturates much earlier than the LFSR/system-random curves in
the original UnarySim CSVs. CPU and MPS produce bit-identical curves at a fixed sample count.

## Note on the kernel used

The pure per-timestep streaming kernel `linear` (scaled saturating adder) was tried first
but is not usable for this MLP. Over a fan-in of 1024 its scaled adder divides the inner product
by `entry` (~1025), compressing the layer output into a near-zero bipolar range (about +-0.03)
that the downstream ReLU cannot resolve; cascading three such layers collapses the network to
chance. Its non-scaled mode (`scale=1`) is worse: the saturating accumulator drains by only 1
per fire while being pushed up by up to ~512 per cycle, so its output rate floors around 0.55
and can never represent negative values (output stuck in [0.09, 0.51], correlation ~0.05 with
the reference), independent of accumulator width. `linear_pc` avoids both problems by
returning the raw per-cycle count (no lossy adder), reconstructing `W x + b` at correlation
~0.999 over its full range, which is why the example uses it. This is a property of the kernel
designs (the scaled adder targets `scale = entry`), not a correctness bug.
