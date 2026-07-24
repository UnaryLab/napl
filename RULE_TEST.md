# NAPL test rules

This file is the canonical verification policy for the Python simulation model only; an RTL-backed change must also satisfy every gate in [RULE_RTL.md](RULE_RTL.md). Every new or changed behavior must have an automated check that fails when the behavior is wrong. Streaming-kernel tests start from [tests/template_streaming_kernel.py](tests/template_streaming_kernel.py) and single-shot trainable-kernel tests from [tests/template_single_shot_trainable.py](tests/template_single_shot_trainable.py); both call the suite drivers `streaming_suite` and `single_shot_suite` in [src/napl/utils/_shared_test.py](src/napl/utils/_shared_test.py), which also holds the shared input cloning, equality checks, and benchmark timing.

## Core gates

1. **Execution model.** A streaming module processes one timestep and increments `timestep_cur` exactly once per call; `reset()` restores the initial state, and replaying the same inputs after reset reproduces the same outputs. A single-shot module sets `streaming = False`, processes the whole tensor in one call, and keeps `timestep_cur == 0`.
2. **Device coverage.** Run the full test body on every device returned by `napl.utils._shared_test.devices()`, moving both the module and the inputs. Skip a device only when the operation lacks support there, and state the reason.
3. **Numerical fidelity.** Include known-answer cases and an analytic reference. Check streaming FSU results against the stochastic-computing bound, normally proportional to `1/sqrt(N)`. Check HUB, FXP, TLUT, and Hard results against the matching PyTorch operation within the stated quantization bound. Use `torch.equal` when integer spike results must be bit-exact; otherwise state and assert the tolerance.
4. **Gradients.** For trainable operations with a straight-through estimator, compare every declared input and parameter gradient with its defining equation. A finite-gradient check alone is not sufficient.
5. **Performance.** Measure the same operation once per device using identical pre-generated CPU inputs. `devices()` returns CPU first; save that result as the baseline and reuse it: `speedup = cpu_runtime / device_runtime`. Use `benchmark` for repeated measurements on one device. For a single device-synchronized elapsed measurement, use `timer`; do not duplicate its synchronization and `perf_counter` logic. Keep analysis and reporting outside the timed block unless they are part of the operation being measured:

   ```python
   from napl.utils._shared_test import timer

   with timer(device) as elapsed:
       operation()
   print(elapsed.seconds)
   ```

Operands that must be independent use distinct Sobol dimensions. Fix or record the random seed and reuse the same logical inputs for the device and CPU runs.

## Verification by change type

- Streaming and single-shot trainable kernels copy the matching template, which enforces every applicable gate via the shared suite drivers.
- Encoder, decoder, metric, or base: boundary timesteps and values, shapes and dtypes, device migration, reset and replay, and public state such as metric `valid` flags.
- Shared utility or public API: focused known-answer and edge-case checks and the affected callers' tests.

The default verification is the focused tests for the changed unit. Run the full sweep (`tests/sweep_test.py`) only when explicitly requested.

## Pass criteria and evidence

Correctness, execution state, reset, and gradient checks are hard gates: they must use assertions and the verification command must exit zero; printed values do not count as a pass. An expected skip must name the unsupported device or feature. Performance evidence is required for every device; the ratio becomes a hard gate only with a stable baseline and a device-specific noise allowance. The suite drivers emit the required evidence (seed, dtype, bound versus observed error, reset and replay, timing statistics); keep their prints.
