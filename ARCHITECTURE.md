# NAPL architecture

This document is the canonical description of NAPL's current design and package boundaries. Verification requirements live in [RULE_SIM.md](RULE_SIM.md) and [RULE_IMP.md](RULE_IMP.md).

## Purpose and dataflow

NAPL is UnaryLab's framework for programmable spike processing (PSP). It uses one-bit spike streams as a common representation for numerical unary computing and neural computing. A tensor is encoded into a stream, processed by spike-domain operations or layers over multiple timesteps, and decoded or measured:

```text
number or tensor
      |
      v
   encode  ->  one-bit spike stream  ->  operation(s) or streaming layer(s)
                                                     |
                                                     v
                                             decode or metric
                                                     |
                                                     v
                                              number or statistic
```

Rate and temporal encodings differ in how the encoder orders spikes. Longer streams generally improve numerical fidelity at the cost of latency.

NAPL also provides non-streaming binary-domain kernels. These process a whole tensor in one call and use quantized fixed-point or floating-point arithmetic rather than an explicit per-timestep stream. Hybrid unary-binary (HUB) wrappers sit between the two paths: they take and return numeric tensors at their boundary codecs and run one-bit spike streams inside, one timestep per inner call. Both execution paths use PyTorch tensors and can run on supported CPU and GPU devices.

The Python model is the source of functional behavior. A subset of spike operations and streaming layers has a Verilog-2001 counterpart under `src/napl/imp/`, verified against Python-generated golden vectors.

## Execution models

### Streaming execution

Streaming classes inherit `napl_base` and use its default `streaming = True`.

- One call to a module represents one timestep and one bit from each input stream.
- `napl_base.__call__` increments `timestep_cur` before `forward()` runs.
- Stateful modules accumulate or update state across calls.
- `reset()` returns `timestep_cur` and module-specific state to their initial values.
- `valid` is true after at least one timestep has run since construction or reset.

The `@napl_sim_timesteps` decorator repeats a module method or free function, whose body represents one timestep. On a `napl_base` module that carries its own run length as `self.timestep`, such as a `*_hub` wrapper, a call with no `timesteps=` keyword is a complete fresh run: the decorator resets the module and its registered children, repeats the body for `timestep` cycles on the same arguments, and returns the final cycle's value. Every other target passes `timesteps=` as a keyword argument, and the decorator then repeats the body that many times without resetting. `napl_base.forward_timestep()` advances a decorated `forward()` one timestep at a time: it ticks once for a streaming module, runs a single timestep of the undecorated body, and never resets. A run counts its cycles on `timestep_cur` for a streaming module; a non-streaming module such as a `*_hub` wrapper keeps its own counter at zero, and the streaming children it drives hold the count.

The usual streaming path is:

```python
spike_a = encoder_a(value_a)
spike_b = encoder_b(value_b)
spike_y = operation(spike_a, spike_b)
value_y = decode(spike_y)
```

Call the full path once per timestep. The encoder uses the current sequence element, operations transform the current spike bits, and the decoder updates its progressive estimate.

### Non-streaming execution

Binary-domain classes and `*_hub` wrappers set `streaming = False`.

- One call processes the complete input tensor.
- `napl_base.__call__` does not increment `timestep_cur`.
- `timestep_cur` stays zero and `valid` remains false.
- Trainable FXP kernels use custom `torch.autograd.Function` implementations and straight-through estimators where needed.

Non-streaming execution is used by FXP linear, convolution, recurrent, and rounding modules, FXP activations, `butterfly_fp`, and the `*_hub` wrappers. It is a simulation model, not an automatic claim that a matching gate-level RTL module exists. A `*_hub` wrapper is non-streaming at its numeric ports and runs spike streams inside: one call encodes the numeric input, drives its streaming core for `timestep` cycles, and decodes the result, so the cycle count shows on the streaming parts it holds rather than on its own `timestep_cur`. The streaming MGU core's gate encoders occupy Sobol number-sequence dimensions `3` to `6` (`_CORE_DIMS` in `sim/module/_shared.py`), so `mgu_hard_mix_hub` requires its input-encoder dimension and its hidden-encoder dimension on the next dimension to stay clear of the gate dimensions `3` to `6`, keeping the port streams decorrelated from the gate weight streams, and its constructor guard raises otherwise.

### Kernel families

| Family | Representation | Execution | Main classes |
| --- | --- | --- | --- |
| Streaming | One-bit spike streams | One timestep per call | `linear_mix`, `conv_mix`, `linear_gaines`, `conv_gaines`, `linear_ugemm`, `conv_ugemm`, `avgpool2d_ugemm`, `mgu_hard_mix`, and streaming operations |
| HUB | Numeric tensors at the boundary codecs, one-bit spike streams inside | Non-streaming at the numeric ports: one complete `timestep`-cycle run per call, one timestep per inner call on the same numeric input | `mgu_hard_mix_hub`, `linear_ugemm_hub`, `conv_ugemm_hub`, `fft_hub`, `fft_dyn_hub` |
| FXP | Quantized fixed-point tensors | One tensor per call | `linear_fxp`, `conv_fxp`, `mgu_hard_fxp`, `round_fxp`, `relu_fxp`, `sigmoid_fxp`, `tanh_fxp` |
| Binary FFT butterfly | Complex numeric tensors | One tensor per call | `butterfly_fp` |
| Streaming FFT butterfly | Spike streams | One timestep per call | `butterfly_ugemm`, `butterfly_ugemm_dyn`, `butterfly_mix`, `butterfly_mix_dyn` |
| Streaming FFT | Spike streams | One timestep per call | `fft`, `fft_dyn` |

The table covers the computational classes. The `sim/metric/` classes `accuracy`, `correlation`, `stability`, `stability_builder`, `stability_flux`, and `stability_norm` are deliberately outside it: they serve stream analysis, measuring the properties of a stream or building a stream that has prescribed ones, instead of computing a result from operand streams. The [metric layer](#metric-layer) section covers them.

A `_dyn` name in the scaling and FFT families marks the variant that takes its scale at call time rather than at construction. Such a class replaces the fixed key `scale` with the ceiling key `scale_max`, in its own config or in a nested one, and each call then supplies the scale in force for that timestep. Exactly six classes follow that rule: `add_scale_dyn`, `div_scale_dyn`, `butterfly_ugemm_dyn`, `butterfly_mix_dyn`, `fft_dyn`, and `fft_dyn_hub`.

`add_scale_dyn` and `div_scale_dyn` hold `scale_max` in their own config, so `napl_base.__init__` both requires it and rejects `scale` as a key outside the accepted list. The other four take it in the nested `add_config`, while their own `mul_config` or `codec_config` accepts the same keys as the static counterpart's. An `add_config` carrying `scale` alone fails their own guard with `Missing key <scale_max> in the dynamic adder configuration`; the `napl_base` rejection of `scale` fires only when both keys are present.

`fft` and `fft_dyn` pick their stage class from the optional `mul_config` key `kernel`: `'ugemm'`, the default, builds the conditional-spike butterflies, and `'mix'` builds the Gaines butterflies, whose stage `index` encodes its constant twiddle on Sobol dimension `5 + index`. `fft_hub` and `fft_dyn_hub` pass `mul_config` to their core unchanged, so they carry the same knob.

## Package map

The Python simulation model lives under `src/napl/sim/` and the hardware tree under `src/napl/imp/`. Public classes and functions live in the six `sim` subpackages (`base`, `module`, `operation`, `metric`, `structure`, `algorithm`) and are imported from the subpackage that defines each name, using the form `from napl.sim.<subpackage> import <name>` (for example `from napl.sim.module import linear_mix` or `from napl.sim.operation import mul_gaines`). `operation` imports only `base` and `utils`, while `module`, `metric`, and `algorithm` import `operation` primitives at file top. `module` and `algorithm` also reach into `metric` for stream measurement. All of these are absolute submodule imports.

Each of the six `sim` subpackages declares `__all__` listing its own public classes. `napl/__init__.py` holds no re-exports: `from napl import <name>` does not resolve, and the flat namespace carries no public classes. `napl.sim` and `napl.utils` stay reachable as submodule attributes, so `from napl.utils import ...` works.

| Path | Responsibility | Main components |
| --- | --- | --- |
| `src/napl/sim/base/` | Shared module lifecycle and global contracts | `napl_base`, timestep decorators, `hw_params`, `pvt_corner`, `timing`, global dtype config |
| `src/napl/sim/module/` | Neural layers | linear, convolution, recurrent, and pooling layers in streaming, FXP, Gaines, uGEMM, and HUB variants |
| `src/napl/sim/operation/` | Stream endpoints and reusable spike and binary primitives | encode, decode; arithmetic, comparison, activation, state, polarity conversion, and stream synchronization |
| `src/napl/sim/metric/` | Progressive stream monitors and stream construction | accuracy, correlation, stability metrics, `stability_builder` |
| `src/napl/sim/algorithm/` | Compositions of modules and operations | binary and streaming FFT butterflies; fixed- and runtime-scale FFTs |
| `src/napl/imp/operation/` | Synthesizable operation counterparts | per-operation RTL, testbenches, and golden-vector generators |
| `src/napl/imp/module/` | Synthesizable module counterparts | lane-replicated module RTL, testbenches, and golden-vector generators |
| `src/napl/imp/` | Hardware build root | shared Makefile and the simulation-to-RTL `mapping.yaml` |
| `src/napl/utils/` | Shared validation and tensor helpers | YAML I/O, config checks, device discovery, random tensors, power-of-two shift shims |
| `src/napl/sim/structure/` | Biological-neuron abstraction boundary | axon, soma, dendrite, synapse, receptor, column placeholders |

`src/napl/main.py` defines the `napl` command-line entry point. The current command only prints a banner.

### Module layer

The module files build tensor-shaped layers and recurrent cells from PyTorch operations and NAPL primitives.

Streaming layers keep computation in the spike domain. Mix, Gaines, and uGEMM variants change the internal multiplication, encoding, and accumulation methods while preserving per-timestep execution. FXP variants operate in the non-streaming domain.

### Operation layer

`operation/encode.py` converts values into spikes by comparing their encoded probability against a generated number sequence, and `operation/decode.py` counts spikes and exposes a progressive decoded value.

The remaining operations are small computational blocks used directly by programs and composed into higher-level modules. The package includes:

- arithmetic and conversion: multiply, add, divide, square, square root, polarity conversion, sign and magnitude;
- comparison and stream control: minimum, maximum, less-than, greater-than, synchronization;
- activation approximations: ReLU, sigmoid, tanh, and exponential variants;
- stateful elements: D flip-flop, JK flip-flop, and shift register.

Each concrete operation owns its Python state and reset behavior. Operations with verified RTL also declare the matching pipeline delay in `self.hw`.

### Metric layer

Metric classes work per timestep. The observers consume the current spike state on each call, while lazy properties expose the accumulated result once the observer becomes valid. `accuracy` tracks the progressive decoded value and its `analyze()` method compares it with a reference. Correlation and stability classes track stream properties. `stability_builder` runs the other direction: each call emits the next spike of a stream designed to hold a requested normalized stability.

## Core contracts

### Configuration and dtypes

NAPL classes accept one `config` dictionary. `napl_base.__init__(config, key_list, optional_key_list, polarity_required)` requires every key in `key_list`, accepts the keys in `optional_key_list` plus `name`, rejects any other key present in the config, and initializes common fields. Common keys include:

- `polarity`: `unipolar` or `bipolar`;
- `timestep`: stream length or simulation horizon;
- `generator`: `sobol`, `lfsr`, `sys`, `rc`, `tc`, `rate`, or `temporal`;
- `dim`: Sobol dimension for a generated stream.

`src/napl/sim/base/global_config.yaml` is the global dtype source. `spike_type` becomes `self.stype`, and `non_spike_type` becomes `self.ntype`. The current defaults are `torch.int8` for spikes and `torch.float32` for non-spike values.

### Polarity and encoding

Unipolar streams represent values in `[0, 1]`. Bipolar streams represent values in `[-1, 1]` by mapping a value `x` to probability `(x + 1) / 2`. Operations that support both domains branch on `self.polarity`, and their expected arithmetic changes with the encoding.

### Stream independence

Independent operands must use independent number sequences. With Sobol encoding, assign distinct `dim` values to operands that must be decorrelated. Reusing a Sobol dimension correlates streams and can bias the result even when each input stream has the correct marginal rate.

Padding is also part of this contract. Bipolar zero-padding in `conv_mix` uses a separate decorrelated rate-0.5 stream because bipolar zero maps to probability 0.5.

### State and reset

Streaming state belongs to the module that updates it. `napl_base.reset()` resets `timestep_cur` and registered child modules before running the subclass `_reset()` hook. A subclass hook therefore restores only its local persistent state to the state used at construction. Python reset state is also the source of truth for RTL reset behavior.

### Hardware timing

`napl_base` initializes `self.hw` as an `hw_params` object:

- `pp_delay` is the technology-independent input-to-output latency in cycles;
- `timing` maps a `pvt_corner` to `timing(cp_delay, ir_delay, or_delay)` values in nanoseconds.

`pp_delay` must match the verified RTL pipeline. It also controls alignment when paths reconverge. Purely combinational operations use `pp_delay = 0` and place their through-delay in `cp_delay`. Registered operations separate internal, input-to-register, and register-to-output timing as defined by `timing`.

The composites named here derive `internal_encode` as the OR over the parts they register: `self.internal_encode = any(part.internal_encode for part in self.children())`, evaluated once the parts are constructed. `linear_ugemm_hub`, `conv_ugemm_hub`, `mgu_hard_mix_hub`, `fft_hub`, `fft_dyn_hub`, `fft`, `fft_dyn`, `butterfly_ugemm`, `butterfly_ugemm_dyn`, `div_iscb`, and `sqrt_traceiscb` write that line, so all five HUB wrappers derive the flag. `div_iscb` and `sqrt_traceiscb` each derive `True` from the one `div_cordiv` kernel they register, whose RTL inlines its own Sobol index; their other parts are all `False`. A composite that derives it reports that its RTL counterpart holds an encoder whenever any part it registers directly holds one. Other composites declare the flag on the class instead, as `linear_ugemm`, `conv_ugemm`, `linear_gaines`, `conv_gaines`, `linear_mix`, `conv_mix`, `mgu_hard_mix`, `butterfly_mix`, and `butterfly_mix_dyn` do. The flag marks an encoder held in the hardware counterpart, covering both encoding that advances conditionally on data and operands such as weights and biases that the counterpart encodes internally from held numeric codes, a registered `encode` instance does not raise it: `mgu_hard_mix_hub` registers two plain Sobol `encode` instances, and with its `mgu_hard_mix` core forced to `False` the hub derives `False` while both encoders stay registered. The binary-domain `conv_fxp`, `linear_fxp`, and `mgu_hard_fxp` hold no encoder and keep the default `False`. `self.children()` is depth-1, so the walk stops at each registered part's own value: the hub reads `True` from `mgu_hard_mix`, which declares `internal_encode = True` on the class because its gate multipliers encode their own operands, and never sees that cell's parts `mul_ugemm` and `mul_ugemm_dyn`, both `True`. Depth-1 covers the whole tree because every class carries its own correct value, declared or derived, so the flag is not transitive; `encode` itself keeps `internal_encode = False`, which keeps a held encoder a statement each holder makes rather than a property inferred from the codec class.

Every operation with generated RTL sets `self.hw.pp_delay`. The non-streaming binary-domain classes `relu_fxp`, `sigmoid_fxp`, `tanh_fxp`, `conv_fxp`, `linear_fxp`, `mgu_hard_fxp`, `round_fxp`, and `butterfly_fp` never set `pp_delay`, so it keeps its default of 0, and they have no gate-level RTL counterpart.

## Hardware boundary

`src/napl/imp/operation/` mirrors concrete operations with implemented RTL. Each operation folder contains its RTL, testbench, Python golden-vector generator, generated vectors, and build output. One Python `forward()` timestep corresponds to one `posedge i_clk`; active-low `i_rst_n` corresponds to Python `reset()`.

Fixed-point accumulators are a simulation-only width. `add_scale`, `add_scale_dyn`, `div_scale`, and `div_scale_dyn` split their signed accumulator into `intwidth` integer bits and `fracwidth` fractional bits and hold it in raw units of `2 ** -fracwidth`, which lets their `scale` be a non-integer quantized to that grid. The RTL accumulators count whole spikes, so only `fracwidth == 0` has a hardware form: `mapping.yaml` rejects any node with a nonzero `fracwidth` for all four, and maps `WIDTH` from `config['intwidth']` wherever the accumulator clamp is reachable. `div_scale_unipolar` and `div_scale_dyn_unipolar` carry no `WIDTH`: their accumulator stays inside the clamp for every configuration the Python constructor accepts, so the width would be unobservable.

`src/napl/imp/module/` mirrors `sim/module` classes whose RTL is a lane replication of operation circuits. Metric and algorithm classes do not currently have matching RTL trees in this repository. Follow [RULE_IMP.md](RULE_IMP.md) for the mandatory design and verification contract, the folder layout and commands, and the module layer's implemented and out-of-scope class lists.

## Incomplete boundaries

| Area | Current state |
| --- | --- |
| Python/PyTorch transpilation | No transpiler package is present in `src/napl/`. |
| CLI | `napl` is a banner-only placeholder. |
| FFT RTL | Binary and streaming FFT implementations are available in Python; the algorithm package has no RTL counterpart. |
| Spike components | `wta` provides temporal earliest-spike selection; `inhibit` provides temporal stream gating. |
| Biological structure | `napl.sim.structure` files are empty placeholders. |

Treat these as package boundaries that are not yet implemented, not as completed interfaces.
