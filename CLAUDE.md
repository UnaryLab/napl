# Agent instructions

This repo contains the code for a research project. `AGENTS.md` is a symlink to this file, so edit `CLAUDE.md` only.

## Project architecture

NAPL ("neuro-adaptive programming language") is UnaryLab's PyTorch framework for programmable spike processing. It supports per-timestep spike-stream simulation and non-streaming binary-domain kernels, with verified Verilog counterparts for a subset of operations. The simulation packages live under `src/napl/sim/` and the RTL tree under `src/napl/imp/`; each public class or function is imported from the subpackage that defines it, as `from napl.sim.<subpackage> import <name>` (e.g. `from napl.sim.module import linear_ugemm`, `from napl.sim.operation import mul_gaines`).

Read [ARCHITECTURE.md](ARCHITECTURE.md) before editing execution semantics, stream encodings, package boundaries, module composition, or the Python-to-RTL contract. It is the canonical source for NAPL's dataflow, execution models, package map, core contracts, and current implementation boundaries.

## Environment & commands

This project uses the `napl` conda env. Always run Python through it: `conda run -n napl python ...`.

- **Install (dev mode):** `conda env create -f environment.yaml && conda activate napl && python3 -m pip install -e . --no-deps`
- **Run the test suite (Python model):** `python tests/sweep_test.py`, which walks `tests/`, runs every `test_*.py` as a standalone script (each test file has an `if __name__ == '__main__'` entry point), logging stderr to `tests/sweep_test.log`. Tests are also valid pytest targets: `conda run -n napl pytest tests/operation/test_mul_gaines.py`.
- **Run a single test directly:** `conda run -n napl python tests/operation/test_mul_gaines.py`
- **CLI:** `napl -h` (entry point `napl.main:main`). The CLI parses the `-sim`/`-syn` modes and their options, then reports that execution is not implemented; it is a placeholder.
- **RTL co-simulation:** from `src/napl/imp/`, `conda run -n napl make test OP=mul_gaines`. Requires Icarus Verilog (`iverilog`/`vvp`) on PATH. See "RTL implementation" below.
- **End-to-end applications (`zoo/`):** currently empty; reserved for applications built on NAPL's public API.
- **Maintenance skills/workflow (`.claude/`):** the repo ships five single-kernel skills (`napl-gen-sim`, `napl-opt-sim`, `napl-gen-rtl`, `napl-validate-unarysim`, `napl-validate-sim-rtl`) and the `napl-port-unarysim` workflow that fans out across them to port, optimize, validate, and lower every class.

### Testing rules

All simulation tests must follow [RULE_SIM.md](RULE_SIM.md), the canonical policy for execution-model checks, device coverage, numerical fidelity, gradients, performance, and verification evidence. RTL-backed changes must also follow [RULE_IMP.md](RULE_IMP.md).

Do not run the full sweep (`tests/sweep_test.py`) without explicit user approval; verify changes with the focused tests for the changed unit.

## Critical implementation gotchas

- **Never use `>>`/`<<` on float tensors.** UnarySim relied on a monkey-patched float bit-shift; stock PyTorch shifts are integer-only. Use the `pow2_lshift`/`pow2_rshift` shims in `utils/utils.py`.
- **Import direction:** `sim/operation` is self-contained and does not import `sim/module`, `sim/metric`, or `sim/algorithm`; `encode`, `decode`, and `gen_num_seq` live in `sim/operation`. `sim/module` imports `sim/operation` primitives at file top. Each `sim` subpackage `__init__.py` star-imports its own members; `napl/__init__.py` holds no re-exports. See [RULE_SIM.md](RULE_SIM.md) gates 11 and 14 for the full import-style and `__init__` export-sorting rules.
- `conv_mix` bipolar zero-padding uses a *decorrelated rate-0.5 pad stream* (a separate pad encoder), not a deterministic 0/1 toggle (which would correlate with the Sobol weight stream).

## RTL implementation (`src/napl/imp/`)

The Verilog-2001 hardware counterpart of `sim/operation/` lives under `src/napl/imp/operation/`, and of `sim/module/` under `src/napl/imp/module/`. Follow [RULE_IMP.md](RULE_IMP.md) for the mandatory RTL design and verification rules plus the layout and commands.

## Git

Don't commit or push unless asked. Generated RTL artifacts are covered by [RULE_IMP.md](RULE_IMP.md).

## Review tiers

Behavior changes under `src/napl/` take the full review with the mutation or bite proof their gates call for. Documentation, comments, README prose, and test message wording take a one-pass spot check, with a verdict of at most three sentences. Mechanical work such as a rename or a formatting pass is verified by the command output the author attaches, which the reviewer confirms and spot-checks rather than repeats: a zero-hit sweep proves the old name is gone, not that the new name is right. Work is mechanical when its correctness is establishable from the command output alone, without reading the diff: renames, formatting passes, and moves or anchor updates qualify. Work whose correctness depends on what the changed code means is not mechanical, however small the diff. The test is whether the attached command can fail in a way that proves the change wrong. The tiers are stated in full in [RULE_SIM.md](RULE_SIM.md) for simulation, [RULE_IMP.md](RULE_IMP.md) for RTL, and [RULE_DOC.md](RULE_DOC.md) for documentation.
