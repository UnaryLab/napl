# Agent instructions

This repo contains the code for a research project. `AGENTS.md` is a symlink to this file, so edit `CLAUDE.md` only.

## Project architecture

NAPL ("neuro-adaptive programming language") is UnaryLab's PyTorch framework for programmable spike processing. It supports per-timestep spike-stream simulation and single-shot binary-domain kernels, with verified Verilog counterparts for a subset of operations. The simulation packages live under `src/napl/sim/` and the RTL tree under `src/napl/hw/`; `napl/__init__.py` re-exports every public class, so `from napl import linear, mul_and` works.

Read [ARCHITECTURE.md](ARCHITECTURE.md) before editing execution semantics, stream encodings, package boundaries, module composition, or the Python-to-RTL contract. It is the canonical source for NAPL's dataflow, execution models, package map, core contracts, and current implementation boundaries.

## Environment & commands

This project uses the `napl` conda env. Always run Python through it: `conda run -n napl python ...`.

- **Install (dev mode):** `conda env create -f environment.yaml && conda activate napl && python3 -m pip install -e . --no-deps`
- **Run the test suite (Python model):** `python tests/sweep_test.py`, which walks `tests/`, runs every `test_*.py` as a standalone script (each test file has an `if __name__ == '__main__'` entry point), logging stderr to `tests/sweep_test.log`. Tests are also valid pytest targets: `conda run -n napl pytest tests/operation/test_mul_and.py`.
- **Run a single test directly:** `conda run -n napl python tests/operation/test_mul_and.py`
- **CLI:** `napl -h` (entry point `napl.main:main`). The CLI currently only prints a banner; it is a placeholder.
- **RTL co-simulation:** from `src/napl/hw/`, `conda run -n napl make test OP=mul_and`. Requires Icarus Verilog (`iverilog`/`vvp`) on PATH. See "RTL implementation" below.
- **End-to-end examples (`examples/`):** three UnarySim app ports, each with its own README: `mlp/` (MLP3 on MNIST in the streaming unary domain: `train_fp.py` then `eval_unary.py --device cpu|mps`), `ubrain/` (uBrain BCI CNN+RNN, FP and HUB forms; `build_hub_from_fp()` shares FP weights into the HUB model), `usystolic/` (uSystolic convnet_mnist; `eval_sweep.py` sweeps HUB over cycles and FXP over bitwidths). Results land in each example's `results/*.csv`.
- **Maintenance skills/workflow (`.claude/`):** the repo ships five single-kernel skills (`napl-gen-sim`, `napl-opt-sim`, `napl-gen-rtl`, `napl-validate-unarysim`, `napl-validate-sim-rtl`) and the `napl-port-unarysim` workflow that fans out across them to port, optimize, validate, and lower every class. Runs append durable ledger rows under `reports/`.

### Testing rules

All tests must follow [RULE_TEST.md](RULE_TEST.md), the canonical policy for execution-model checks, device coverage, numerical fidelity, gradients, performance, and verification evidence. RTL-backed changes must also follow [RULE_RTL.md](RULE_RTL.md).

## Critical implementation gotchas

- **Never use `>>`/`<<` on float tensors.** UnarySim relied on a monkey-patched float bit-shift; stock PyTorch shifts are integer-only. Use the `pow2_lshift`/`pow2_rshift` shims in `utils/utils.py`.
- **Import cycle:** `sim/module/{linear,conv,rnn}` must import `sim/operation` primitives **lazily inside `__init__`**, because `sim/operation/{mul_csg,tanh_p1,exp_n1}` import `sim/module/encoder` (and `sim/operation/{add_gaines,div_cordiv,div_gaines,sqrt_gaines}` import `sim/module`) at import time. `sim/module/__init__` imports `linear` before `conv`.
- `conv` bipolar zero-padding uses a *decorrelated rate-0.5 pad stream* (a separate pad encoder), not a deterministic 0/1 toggle (which would correlate with the Sobol weight stream).

## RTL implementation (`src/napl/hw/`)

The Verilog-2001 hardware counterpart of `sim/operation/` lives under `src/napl/hw/`. Follow [RULE_RTL.md](RULE_RTL.md) for the mandatory RTL design and verification rules plus the layout and commands.

## Git

Don't commit or push unless asked. Generated RTL artifacts are covered by [RULE_RTL.md](RULE_RTL.md).
