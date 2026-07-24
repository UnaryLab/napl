# NAPL architecture

This document is the canonical description of NAPL's current design and package boundaries. Verification requirements live in [RULE_TEST.md](RULE_TEST.md) and [RULE_RTL.md](RULE_RTL.md).

## Purpose and dataflow

NAPL is UnaryLab's framework for programmable spike processing (PSP). It uses one-bit spike streams as a common representation for numerical unary computing and neural computing. A tensor is encoded into a stream, processed by spike-domain operations or layers over multiple timesteps, and decoded or measured:

```text
number or tensor
      |
      v
   encoder  ->  one-bit spike stream  ->  operation(s) or streaming layer(s)
                                                     |
                                                     v
                                             decoder or metric
                                                     |
                                                     v
                                              number or statistic
```

Rate and temporal encodings differ in how the encoder orders spikes. Longer streams generally improve numerical fidelity at the cost of latency.

NAPL also provides single-shot binary-domain kernels. These process a whole tensor in one call and use quantization or hybrid unary-binary arithmetic rather than an explicit per-timestep stream. Both execution paths use PyTorch tensors and can run on supported CPU and GPU devices.

The Python model is the source of functional behavior. A subset of spike operations has a Verilog-2001 counterpart under `src/napl/hw/`, verified against Python-generated golden vectors.

## Execution models

### Streaming execution

Streaming classes inherit `napl_base` and use its default `streaming = True`.

- One call to a module represents one timestep and one bit from each input stream.
- `napl_base.__call__` increments `timestep_cur` before `forward()` runs.
- Stateful modules accumulate or update state across calls.
- `reset()` returns `timestep_cur` and module-specific state to their initial values.
- `valid` is true after at least one timestep has run since construction or reset.

The `@napl_sim_timesteps` decorator repeats a module method, and `napl_sim_timesteps_func` repeats a free function. Both require `timesteps=` as a keyword argument. The decorated body still represents one timestep.

The usual streaming path is:

```python
spike_a = encoder_a(value_a)
spike_b = encoder_b(value_b)
spike_y = operation(spike_a, spike_b)
value_y = decoder(spike_y)
```

Call the full path once per timestep. The encoder uses the current sequence element, operations transform the current spike bits, and the decoder updates its progressive estimate.

### Single-shot execution

Binary-domain classes set `streaming = False`.

- One call processes the complete input tensor.
- `napl_base.__call__` does not increment `timestep_cur`.
- `timestep_cur` stays zero and `valid` remains false.
- Trainable HUB, FXP, TLUT, and hard neural kernels use custom `torch.autograd.Function` implementations and straight-through estimators where needed.

Single-shot execution is used by binary linear and convolution layers, binary recurrent cells, HUB activations, and fixed-point rounding. It is a simulation model, not an automatic claim that a matching gate-level RTL module exists.

### Kernel families

| Family | Representation | Execution | Main classes |
| --- | --- | --- | --- |
| Streaming | One-bit spike streams | One timestep per call | unsuffixed `sim/module/` classes (`linear`, `conv`, `mgu`, ...), `sim/operation/` |
| HUB | Binary tensors with hybrid unary-binary arithmetic | One tensor per call | `linear_hub`, `conv_hub`, `mgu_hub`; HUB activations in `sim/operation/` |
| FXP | Quantized fixed-point tensors | One tensor per call | `linear_fxp`, `conv_fxp`, `mgu_hardfxp`, `round_fxp` |
| TLUT | Table-based quantized tensors | One tensor per call | `linear_tlut`, `conv_tlut` |
| Hard neural cells | Binary tensors with hard nonlinearities | One tensor per call | `mgu_hard*`, `gru_hard*` |

## Package map

The Python simulation model lives under `src/napl/sim/` and the hardware tree under `src/napl/hw/`. `napl/__init__.py` star-imports the six `sim` subpackages (`base`, `module`, `operation`, `metric`, `structure`, `algorithm`), so every public class is importable at the top level: `from napl import linear, mul_and, accuracy, napl_base`. Deep imports such as `from napl.sim.operation import mul_and` also work.

| Path | Responsibility | Main components |
| --- | --- | --- |
| `src/napl/sim/base/` | Shared module lifecycle and global contracts | `napl_base`, timestep decorators, `hw_params`, `pvt_corner`, `timing`, global dtype config |
| `src/napl/sim/module/` | Stream endpoints and neural layers | encoder, decoder; linear, convolution, recurrent, and pooling layers in streaming, HUB, FXP, TLUT, Gaines, and uGEMM variants |
| `src/napl/sim/operation/` | Reusable spike and binary primitives | arithmetic, comparison, activation, state, polarity conversion, and stream synchronization |
| `src/napl/sim/metric/` | Progressive stream monitors and stream construction | accuracy, correlation, stability metrics, `stability_builder` |
| `src/napl/sim/algorithm/` | Compositions of modules and operations | FFT butterfly |
| `src/napl/hw/` | Synthesizable hardware counterparts | per-operation RTL, testbenches, golden-vector generators, shared Makefile |
| `src/napl/utils/` | Shared validation and tensor helpers | YAML I/O, config checks, device discovery, random tensors, power-of-two shift shims |
| `src/napl/sim/structure/` | Biological-neuron abstraction boundary | axon, soma, dendrite, synapse, receptor, column placeholders |

`src/napl/main.py` defines the `napl` command-line entry point. The current command only prints a banner.

### Module layer

`module/encoder.py` converts values into spikes by comparing their encoded probability against a generated number sequence. `module/decoder.py` counts spikes and exposes a progressive decoded value. The remaining module files build tensor-shaped layers and recurrent cells from PyTorch operations and NAPL primitives.

Streaming layers keep computation in the spike domain. Partial-count and uGEMM variants change the internal accumulation method while preserving per-timestep execution. Gaines linears select combinations of Gaines multiplication and addition. HUB, FXP, TLUT, and hard variants operate in the single-shot domain.

### Operation layer

Operations are small computational blocks used directly by programs and composed into higher-level modules. The package includes:

- arithmetic and conversion: multiply, add, divide, square, square root, polarity conversion, sign and magnitude;
- comparison and stream control: minimum, maximum, less-than, greater-than, synchronization;
- activation approximations: ReLU, sigmoid, tanh, and exponential variants;
- stateful elements: D flip-flop, JK flip-flop, and shift register.

Each concrete operation owns its Python state and reset behavior. Operations with verified RTL also declare the matching pipeline delay in `self.hw`.

### Metric layer

Metrics are streaming observers. Each call consumes the current spike state, while lazy properties expose the accumulated result after the observer becomes valid. `accuracy` tracks the progressive decoded value and its `analyze()` method compares it with a reference. Correlation and stability classes track stream properties.

## Core contracts

### Configuration and dtypes

NAPL classes accept one `config` dictionary. `napl_base.__init__(config, key_list, polarity_required)` validates required keys and initializes common fields. Common keys include:

- `polarity`: `unipolar` or `bipolar`;
- `timestep`: stream length or simulation horizon;
- `generator`: `sobol`, `lfsr`, `sys`, `rc`, `tc`, `rate`, or `temporal`;
- `dim`: Sobol dimension for a generated stream.

`src/napl/sim/base/global_config.yaml` is the global dtype source. `spike_type` becomes `self.stype`, and `non_spike_type` becomes `self.ntype`. The current defaults are `torch.int8` for spikes and `torch.float32` for non-spike values.

### Polarity and encoding

Unipolar streams represent values in `[0, 1]`. Bipolar streams represent values in `[-1, 1]` by mapping a value `x` to probability `(x + 1) / 2`. Operations that support both domains branch on `self.polarity`, and their expected arithmetic changes with the encoding.

### Stream independence

Independent operands must use independent number sequences. With Sobol encoding, assign distinct `dim` values to operands that must be decorrelated. Reusing a Sobol dimension correlates streams and can bias the result even when each input stream has the correct marginal rate.

Padding is also part of this contract. Bipolar zero-padding in `conv` uses a separate decorrelated rate-0.5 stream because bipolar zero maps to probability 0.5.

### State and reset

Streaming state belongs to the module that updates it. `napl_base.reset()` resets `timestep_cur` and registered child modules before running the subclass `_reset()` hook. A subclass hook therefore restores only its local persistent state to the state used at construction. Python reset state is also the source of truth for RTL reset behavior.

### Hardware timing

`napl_base` initializes `self.hw` as an `hw_params` object:

- `pp_delay` is the technology-independent input-to-output latency in cycles;
- `timing` maps a `pvt_corner` to `timing(cp_delay, ir_delay, or_delay)` values in nanoseconds.

`pp_delay` must match the verified RTL pipeline. It also controls alignment when paths reconverge. Purely combinational operations use `pp_delay = 0` and place their through-delay in `cp_delay`. Registered operations separate internal, input-to-register, and register-to-output timing as defined by `timing`.

Every operation with generated RTL sets `self.hw = hw_params(pp_delay=...)`. Only `tanh_hub`, `sigmoid_hub`, and `round_fxp` carry a scalar `self.delay` instead; they are single-shot classes with no gate-level RTL counterpart.

## Hardware boundary

`src/napl/hw/` mirrors only concrete operations with implemented RTL. Each operation folder contains its RTL, testbench, Python golden-vector generator, generated vectors, and build output. One Python `forward()` timestep corresponds to one `posedge i_clk`; active-low `i_rst_n` corresponds to Python `reset()`.

Higher-level linear, convolution, recurrent, metric, and algorithm classes do not currently have matching RTL trees in this repository. Follow [RULE_RTL.md](RULE_RTL.md) for the mandatory design and verification contract and the folder layout and commands.

## Incomplete boundaries

| Area | Current state |
| --- | --- |
| Python/PyTorch transpilation | No transpiler package is present in `src/napl/`. |
| CLI | `napl` is a banner-only placeholder. |
| FFT | `butterfly` is implemented; `fft` is a placeholder. |
| Spike components | `wta` and `inhibit` are placeholders. |
| Biological structure | `napl.sim.structure` files are empty placeholders. |
| RTL coverage | Only the concrete operation folders under `src/napl/hw/` have hardware counterparts. |

Treat these as package boundaries that are not yet implemented, not as completed interfaces.
