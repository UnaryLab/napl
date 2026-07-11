# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What NAPL is

NAPL ("neuro-adaptive programming language") is UnaryLab's framework for programmable spike processing (PSP), which treats the spike as a single substrate serving both numerical computing (unary computing) and neural computing (neuromorphic computing). It provides a PyTorch-based interface for functional simulation and lowers the same program to hardware. 
In spike processing, a binary value is encoded as a serial stream of 1-bit spikes, computed bit-by-bit through gate-level circuits over many timesteps, then decoded back to binary. 
The two primary stream encodings are rate and temporal, each used on the numerical (unary computing) and the neural (neuromorphic computing) side. Accuracy generally improves with the timestep count, trading latency for precision.
There are two ways into NAPL: write NAPL programs directly, or write conventional Python/PyTorch and have it transpiled to a NAPL program. 
Either path runs as a functional simulation on CPU or GPU, and can be turned into Verilog RTL with identical functionality and predictable timing.
NAPL provides two kernel families: fully streaming unary (fsu) kernels, which operate natively in the spike domain for efficiency, and hybrid unary-binary (hub) kernels, which process a whole binary tensor at once. 
The goal is to let users write high-level PyTorch once, run it efficiently on CPU and GPU, and generate a matching hardware implementation with predictable timing.

## Environment & commands

This project uses the `napl` conda env. Always run Python through it: `conda run -n napl python ...` (a bare `python` on PATH is likely the wrong interpreter).

- **Install (dev mode):** `conda env create -f environment.yaml && conda activate napl && python3 -m pip install -e . --no-deps`
- **Run the test suite (Python model):** `python tests/sweep_test.py`, which walks `tests/`, runs every `test_*.py` as a standalone script (each test file has an `if __name__ == '__main__'` entry point), logging stderr to `tests/sweep_test.log`. Tests are also valid pytest targets: `conda run -n napl pytest tests/operation/test_mul_and.py`.
- **Run a single test directly:** `conda run -n napl python tests/operation/test_mul_and.py`
- **CLI:** `napl -h` (entry point `napl.main:main`). The CLI currently only prints a banner; it is a placeholder.
- **RTL co-simulation:** from `src/napl/implementation/`, `conda run -n napl make test OP=mul_and`. Requires Icarus Verilog (`iverilog`/`vvp`) on PATH. See "RTL implementation" below.
- **End-to-end examples (`examples/`):** three UnarySim app ports, each with its own README: `mlp/` (MLP3 on MNIST in the FSU unary domain: `train_fp.py` then `eval_unary.py --device cpu|mps`), `ubrain/` (uBrain BCI CNN+RNN, FP and HUB forms; `build_hub_from_fp()` shares FP weights into the HUB model), `usystolic/` (uSystolic convnet_mnist; `eval_sweep.py` sweeps HUB over cycles and FXP over bitwidths). Results land in each example's `results/*.csv`.
- **Maintenance skills/workflow (`.claude/`):** the repo ships five single-kernel skills (`napl-gen-sim`, `napl-opt-sim`, `napl-gen-rtl`, `napl-validate-unarysim`, `napl-validate-sim-rtl`) and the `napl-port-unarysim` workflow that fans out across them to port, optimize, validate, and lower every class. Runs append durable ledger rows under `reports/`.

### Testing rules

Every test must run on **both CPU and GPU** and cover **both correctness and performance**.

- **Both devices, every time.** Parametrize over every available device, not just CUDA. On Apple silicon the GPU is **MPS** (`torch.backends.mps.is_available()`); the common `device = 'cuda' if torch.cuda.is_available() else 'cpu'` idiom silently skips the GPU on a Mac. Build the device list as `['cpu'] + (['cuda'] if torch.cuda.is_available() else []) + (['mps'] if torch.backends.mps.is_available() else [])` and run the test body on each. Move both the module (`.to(device)`) and the inputs. Skip a device only if an op genuinely lacks support there, and log why.
- **Numerical (functionality) test.** Per device, apply the fidelity criterion: streaming (FSU) kernels vs the analytic reference within the SC bound (~1/sqrt(N)); binary-domain HUB/FXP/TLUT/Hard vs `nn.Linear`/`nn.Conv2d` within the quant bound; plus known-answer checks. For integer-valued spike operands the kernels are bit-exact across devices (`torch.equal`); assert exactness where it holds, a tolerance otherwise.
- **Performance (speedup) test.** Per device, time the kernel against its baseline (previous implementation or the obvious reference path) and report the ratio. Two non-negotiables: **synchronize before timing on GPU** (`torch.mps.synchronize()` / `torch.cuda.synchronize()`) because GPU execution is async, and **feed identical inputs to every variant being compared** (regenerating random inputs per run silently compares different data and invalidates both the speedup and the correctness check).

## Core architecture

Everything is a `napl_base` (`base/base.py`), a `torch.nn.Module` subclass that loads the global dtype config and tracks `timestep_cur`. Two execution paradigms coexist:

1. **Streaming / per-timestep (the unary-computing core).** A spike circuit is described in a `forward()` that processes **one timestep** (one bit of every stream). Calling it `timesteps` times accumulates the result. The `@napl_sim_timesteps` decorator (and `napl_sim_timesteps_func` for free functions) wraps a `forward()` to run it N times; it **requires** a `timesteps=` kwarg. Each module advances state via `self.tick()` (increments `timestep_cur`) and clears it via `reset()`. The canonical round-trip is **encoder, operation(s), decoder**, then `metric.report_error(decoder.spike_value, reference)`. See `tests/operation/test_mul_and.py` for the template wiring.

2. **Single-shot binary-domain.** HUB/FXP/TLUT and Hard-MGU cells (in `module/linear`, `conv`, `rnn`) are `napl_base` subclasses whose `forward()` processes the **whole tensor at once** (no `tick()`, `timestep_cur` stays 0) and are trainable via `torch.autograd.Function` + straight-through estimators. These are tested directly against `nn.Linear`/`nn.Conv2d`.

### Layers (`src/napl/`)

- **`base/`**: `napl_base`, the timestep decorators, and `global_config` loaded from `base/global_config.yaml`. Config sets `spike_type` (default `torch.int8`; legal: float/bfloat16/int8) and `non_spike_type` (default `torch.float32`); these become `self.stype` / `self.ntype` everywhere. Editing the YAML changes simulation dtypes globally.
- **`module/`**: `encoder` (number to spike stream via comparison against an RNG `num_seq`), `decoder` (spike stream to number by counting), and neural layers `linear`/`conv`/`rnn` (FSU streaming kernels `*_fsu`, the partial-count `*_fsu_pc` streaming variant in `conv_fsu_pc`/`linear`, plus binary-domain `*_hub`/`*_fxp`/`*_tlut`/`*_hard` variants). `wta` is a placeholder.
- **`operation/`**: the gate-level primitives, namely `mul` (`mul_and`=AND/XNOR, `mul_csg`), `add`, `div`, `sqrt`, `square`, `compare` (min/max/lt/gt), activations (`relu`/`sigmoid`/`tanh`), stateful elements (`dff`/`jkff`/`shiftreg`), polarity converters (`bi2uni`/`uni2bi`), and more.
- **`metric/`**: `accuracy` (progressive error / `report_error`), `correlation`, `stability`.
- **`algorithm/fft/`**: `butterfly` is implemented; `fft` is a placeholder.
- **`utils/`**: shared helpers, namely YAML I/O, config/polarity/name checks used by `napl_base.__init__`, random tensor generation, and `pow2_lshift`/`pow2_rshift` (see gotchas).
- **`structure/`**: biological-neuron abstraction (axon/soma/dendrite/synapse/receptor/column). **All placeholders**, and `structure/__init__` imports a nonexistent `minicolumn`, so `import napl.structure` currently fails. Treat empty modules as "not yet built," not bugs.

### Key conventions

- Modules take a single `config` dict; `napl_base.__init__(config, key_list, polarity_required)` validates required keys. Common keys: `polarity` (`'unipolar'`/`'bipolar'`), `timestep`, `generator` (`sobol`/`lfsr`/`sys`/`rc`/`tc`/`rate`/`temporal`), `dim` (Sobol dimension).
- **Decorrelation matters:** operands that must be independent are encoded on **distinct Sobol dims** (see the two encoders in `test_mul_and.py`). Reusing a dim correlates streams and biases results.
- **Polarity:** unipolar maps value to [0,1]; bipolar maps value to [-1,1] via `prob=(x+1)/2`. Many ops branch on `self.polarity`.
- **Hardware contract (`self.hw`, a `hw_params` from `base/base.py`):** each op sets `self.hw = hw_params(pp_delay=...)` in `__init__`. `pp_delay` = input→output latency in cycles (0 = combinational); it is the one field that must match the generated RTL for sim/HW timing to agree. Per-corner STA numbers live in `self.hw.timing`, a `{pvt_corner: timing}` dict (key = MCMM scenario `node/process/voltage/temp/rc/mode`; value = `cp_delay`/`ir_delay`/`or_delay` in ns). Combinational ops (`pp_delay=0`) put the whole through-delay in `cp_delay` with `ir_delay=or_delay=0`; registered ops fill all three. **Migration nearly complete:** this supersedes the scalar `self.delay`. Every op with generated RTL now sets `self.hw = hw_params(pp_delay=...)` (pp_delays were written back from the verified RTL); only the single-shot HUB/hard activation ops with no gate-level RTL (`tanh`, `round`, `sigmoid`) still carry `self.delay`. `implementation/README.md` and the `napl-gen-rtl` skill still mention `self.delay`.

### Gotchas (non-obvious)

- **Never use `>>`/`<<` on float tensors.** UnarySim relied on a monkey-patched float bit-shift; stock PyTorch shifts are integer-only. Use the `pow2_lshift`/`pow2_rshift` shims in `utils/utils.py`.
- **Import cycle:** `module/{linear,conv,rnn}` must import `operation` primitives **lazily inside `__init__`**, because `operation/mul` imports `module/encoder` at import time. `module/__init__` imports `linear` before `conv`.
- `conv_fsu` bipolar zero-padding uses a *decorrelated rate-0.5 pad stream* (a separate pad encoder), not a deterministic 0/1 toggle (which would correlate with the Sobol weight stream).

## RTL implementation (`src/napl/implementation/`)

Verilog-2001 hardware counterpart of `operation/`: one synthesizable module per concrete operation variant, **verified against the Python model** via golden-vector co-simulation (expected outputs always come from the napl Python model, never a hand-written truth table). ~26 ops are implemented (`mul_and`, `mul_csg`, `add_any`, `div_cordiv`, `div_iscb`, `sqrt_emit`, `sqrt_trace*`, compare ops `gt/lt/min/max_*`, `relu_cnt`/`relu_sat`, `sigmoid_hard`, `tanh_hard`, `dff`/`jkff`/`shiftreg`, `bi2uni`/`uni2bi`, `sign_abs`, `square_dff`, `sync_skewed`); `mul_and` (unipolar AND / bipolar XNOR) is the canonical example. Per-op layout: `<op>/{rtl,tb,gen,vec,build}/`. `_gen_common.py` is the shared helper for golden-vector generators (`encode_value`, `pair_streams`, ...) so RTL is driven with the exact spike streams the op's `test_<op>.py` produces. `make test OP=<op>` runs `gen/gen_<op>.py` to emit vectors from the Python model, compiles `rtl/*.v` + the testbench with `iverilog`, and simulates with `vvp`; the testbench prints `PASS` only on a full match. Conventions: RTL follows the global Verilog rules auto-injected by the `verilog-rules.sh` hook when editing `.v` files (Verilog-2001, clock `i_clk`, active-low reset `i_rst_n`, `i_`/`o_` port prefixes, one module per file with filename == module). One module per concrete variant (no parameter-selected variants), named `<op>` with at most a single `_unipolar`/`_bipolar` polarity postfix; one Python `forward()` timestep == one `posedge i_clk`, and active-low `i_rst_n` maps to Python `reset()`. See `src/napl/implementation/README.md` for the full spec.

## Git

Don't commit or push unless asked. `vec/*.vec` and `build/` under `implementation/` are generated artifacts (gitignored).
