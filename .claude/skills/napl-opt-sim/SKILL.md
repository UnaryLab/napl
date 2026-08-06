---
name: napl-opt-sim
description: >-
  Optimize a napl kernel/op/module's functional simulation for speed and report the
  measured CPU and GPU speedup, gated on its test_<kernel>.py still passing. Use
  whenever the user wants a napl op/kernel/layer to run faster ("speed up mul_gaines",
  "make conv faster on GPU", "optimize add_any", "why is sqrt_emit slow", "profile
  and accelerate linear"), wants a before/after speedup number on CPU and/or
  MPS/CUDA, or asks to accelerate the napl simulation without changing results, even
  without the word "optimize". Behavior-preserving only: numerical outputs must not
  change, and the speedup is measured against the kernel's own pre-optimization runtime
  on identical inputs. Sibling to napl-validate-unarysim and the napl-port-unarysim
  workflow (the batch version of this single-kernel skill). Every run leaves a durable
  result row in reports/napl-opt-sim-report.md (kernel, changed?, test gate, CPU/GPU
  before-after speedup).
---

# Optimize a napl kernel and report its CPU/GPU speedup

## Scope

napl is a functional simulator of spike processing; the same `forward()` runs over many
timesteps on CPU and GPU, so per-op throughput is the bottleneck for any real workload.
This skill makes one kernel **faster without changing what it computes**, and proves it
with a before/after number on every device. It is the single-kernel, interactive
counterpart to the **napl-port-unarysim** workflow's Improve phase (which sweeps every class
in a batch). It borrows that phase's discipline and napl-validate-unarysim's measurement
rigor, distilled into one rule each:

- **From napl-port-unarysim/Improve:** the edit is behavior-preserving (numerical outputs MUST
  NOT change; a speedup that changes a result is a bug), follows `coding-discipline`, and
  is gated by the kernel's committed test.
- **From napl-validate-unarysim:** generate inputs ONCE and feed the *same* tensors to the
  before and after runs; time on **both CPU and GPU**; **synchronize the GPU before
  stopping the clock** (async execution otherwise times the launch, not the compute).

## Handoffs

- `coding-discipline`: invoked first in Step 3 as the coding overlay for the edit (think
  before coding, simplicity first, surgical changes, verifiable success criteria).
- `napl-validate-unarysim`: sibling skill with the same measurement rigor, applied to
  correctness-vs-UnarySim instead of speed. Reuse its test-wiring import pattern.
- `napl-gen-rtl`: sibling on the RTL side (lowers a napl kernel to Verilog).
- The `napl-port-unarysim` workflow's Improve phase is the batch sweep this skill is the
  interactive, single-kernel version of.

## Persona

A careful optimizer who baselines before and after on identical inputs, never lets a
speedup change a numerical result, gates every edit on the kernel's committed test, and
reports honestly, including a regression on one device or a no-change (~1.0x) outcome.

## Inputs

- The napl kernel/op/module target to speed up (e.g. `operation.mul_gaines`,
  `module.conv`, `operation.add_any`).
- Run everything through the project env: `conda run -n napl python ...` (a bare `python`
  is the wrong interpreter). Heredocs piped through `conda run` swallow stdout, so write a
  `.py` file and run it, never inline a `python - <<EOF`.
- Time on **every available device**, not just one. Build the list with the bundled helper
  `bench.device_list()` (`['cpu']` + `cuda` if present + `mps` if present). The committed
  tests use the `device = 'cuda' if torch.cuda.is_available() else 'cpu'` idiom, which
  **silently skips MPS on a Mac**, so do not copy that; the harness here must cover MPS.

## Output contract

A verdict: what changed, the test-gate result, and a per-device speedup table (baseline
ms, optimized ms, ratio) measured on identical inputs, plus a durable row in
`reports/napl-opt-sim-report.md` (kernel, changed?, test gate, CPU/GPU before-after
speedup).

## Workflow

### Step 0 - Always run the optimization in a subagent

**Always delegate the optimize-and-time job to one subagent; never run it in the main
thread.** It is a self-contained, multi-step job (read the kernel + its test, baseline
the timing, edit, run the test gate, re-time, interpret) that emits a lot of transient
output: profiling, timing noise, edit iterations, test logs. Keeping that out of the main
conversation is the point. When this skill is invoked with a kernel target, the main
agent's job is only to dispatch and relay:

1. Spawn **one** `general-purpose` subagent (the `Agent`/`Task` tool) with a prompt like:

   > Optimize the napl kernel `<target>` for CPU and GPU speed. Follow the skill at
   > `.claude/skills/napl-opt-sim/SKILL.md` exactly: read it, then execute Steps 1-6:
   > resolve the kernel + its `tests/.../test_<kernel>.py`, baseline its CPU/MPS runtime
   > on identical inputs, make behavior-preserving speedups under `coding-discipline`,
   > gate on the test still passing on every device, re-time, report the per-device
   > speedup table, then record the result row with `scripts/record_opt.py`. Return the
   > verdict table, the diff summary, and the recorded log row.

2. When it returns, relay the verdict (speedup table + what changed + gate result) to the
   user. If the subagent reports the test no longer passes, surface that it reverted (or
   should revert) rather than presenting an unsafe speedup.

**If you ARE that dispatched subagent** (you were handed a specific kernel and told to
follow this skill), ignore this section and execute Steps 1-6 directly. Optimize one
kernel per subagent.

### Step 1 - Resolve the kernel and its test

Pin down the exact napl class (e.g. `operation.mul_gaines`, `module.conv`,
`operation.add_any`) and its source file under `src/napl/`. Its workload and correctness
gate both come from `tests/<subpackage>/test_<kernel>.py` (e.g.
`tests/operation/test_mul_gaines.py`). Read that test: it defines the canonical wiring
(encode(s) -> op -> decoder over timesteps), the configs, and the input construction you
will reuse. Most op/kernel tests expose a `napl_<kernel>` module class plus a
`test_<kernel>()` function; some (binary-domain `*_hub`/`*_fxp`/`*_tlut`/`*_hard`) inline
their wiring instead.

### Step 2 - Baseline the timing (BEFORE any edit)

Measure the current kernel first, so the speedup is real and reproducible. Write a small
timing script `/tmp/time_<kernel>.py` that:

- imports the bundled helpers: `import sys; sys.path.insert(0, '.claude/skills/napl-opt-sim/scripts'); from bench import device_list, time_ms`
- imports the test's wiring and **mirrors its exact configs and input construction** (so
  the workload is the same code the test exercises; see napl-validate-unarysim for the
  `importlib.util.spec_from_file_location` pattern to pull a `napl_<kernel>` class out of
  the test file). For an inlined test, replicate its config + call sequence.
- generates the inputs **once** under a fixed `torch.manual_seed(0)` so a later run on the
  optimized code consumes identical data.
- for each `device_list()` device: move the module (`.to(device)`) and inputs to it, then
  `time_ms(thunk, device)` where `thunk` runs one full workload from a clean state (call
  `module.reset()` then the `timesteps` loop). Print the per-device ms as JSON.

Run it once and record the per-device baseline ms. Capture them now, since the same script
is rerun verbatim in Step 5.

### Step 3 - Optimize (behavior-preserving)

FIRST invoke the **`coding-discipline`** skill (Skill tool) and follow it: think before
coding, simplicity first, surgical changes, verifiable success criteria. Then speed up the
kernel for CPU and GPU. Common wins in this codebase: vectorize per-timestep Python loops
into tensor ops, kill redundant `.clone()`/dtype casts/allocations in the hot path, replace
an elementwise spike-product reduction with a `matmul`, hoist invariants out of the
timestep loop, avoid `torch.roll`-style full-buffer reallocation (see the circular-buffer
idiom in `shiftreg`/`dff`). `decoder` uses the lazy-readout pattern in
`src/napl/sim/metric/accuracy.py:50-75` and `src/napl/sim/operation/decoder.py:30-58`:
`forward()` accumulates only and the on-demand `spike_value` property computes the readout,
giving a 2.28x isolated CPU speedup. For another class, if adopting this pattern changes the
`forward()` return contract, surface it as an explicit API task and never apply it silently.

The hard constraints that keep the edit behavior-preserving (a violation is a bug, not a
speedup) are in **Rules** below; if there is no safe speedup, change nothing and say so
(report ~1.0x).

### Step 4 - Gate on the test (correctness)

Run the kernel's committed test on **every device** and require it to pass before trusting
any speedup:

```bash
conda run -n napl pytest tests/<subpackage>/test_<kernel>.py
```

(also valid as a script: `conda run -n napl python tests/<subpackage>/test_<kernel>.py`).
The test embeds the fidelity criterion (SC ~1/sqrt(N) for streaming kernels, the quant
bound for binary-domain, bit-exact where it holds) plus its known-answer checks, so a pass
is the correctness contract. If it fails, the optimization is invalid: **revert** and
either try a different approach or report that no safe speedup was found. Note the MPS
caveat from Inputs: if the test's own device idiom skips MPS, also exercise the
kernel on MPS via the Step 2 harness so the optimization is proven there too.

**Shared primitives need a downstream gate.** Many ops compose others (e.g. `add_any` is
imported by `relu`/`sigmoid`/`sqrt`/`linear`/`conv`/`rnn`; `sync_skewed` by `div`/`compare`;
`dff` by `square`). A behavior-preserving edit to such a primitive must not break the
kernels that build on it, and `test_<kernel>.py` alone will not catch that. Check whether
the kernel is a dependency: `rg "import .*\b<kernel>\b" src/napl/{operation,module}`. If it
has dependents, also run their tests, or just run the whole independent sweep the way the
napl-port-unarysim Improve phase does (it exists for exactly this cross-file breakage):

```bash
conda run -n napl python tests/sweep_test.py   # exits non-zero and lists the failing files; log goes to tests/sweep_test.log
```

The sweep prints one `Running: <path>` per test file, exits zero when all pass, and on any
failure exits non-zero and prints each failing file; read `tests/sweep_test.log` for the
tracebacks. A downstream failure invalidates the optimization just as the kernel's own test
would.

### Step 5 - Re-time and report the speedup

Rerun the **same** `/tmp/time_<kernel>.py` (fresh process, so it imports the edited code;
the fixed seed reproduces identical inputs) to get the optimized per-device ms. Compute
`speedup = baseline_ms / optimized_ms` per device. Deliver the verdict:

```
Kernel: operation.<name>   Test gate: PASS (cpu, mps)
Changed: <one-line summary of the edit + why it is faster>

| device | baseline (ms) | optimized (ms) | speedup |
|--------|---------------|----------------|---------|
| cpu    | 93.2          | 41.8           | 2.23x   |
| mps    | 53.4          | 22.1           | 2.42x   |
```

Report honestly: if a device got no faster (or slower), show it. A regression on one
device while another improves is a real finding, not something to hide. If nothing was
changed, say so and report 1.0x rather than inventing a number.

### Step 6 - Record the result (always)

Every run must leave a durable, trackable row in **`reports/napl-opt-sim-report.md`** so the
kernels' optimization history is tracked over time (the same "record always" discipline
napl-validate-unarysim and napl-validate-sim-rtl apply to their logs). Append the row with the
bundled recorder (it creates the file with a header on first use, dedupes, and keeps rows sorted
by kernel):

Also pass `--source` (the kernel's repo-relative source file) and `--hash` (its
`git hash-object <source>` value AFTER any edits): these double as the idempotency key the
napl-port-unarysim workflow reads, so a later sweep skips a file whose hash is unchanged.

```bash
conda run -n napl python .claude/skills/napl-opt-sim/scripts/record_opt.py \
    --kernel operation.add_any \
    --source src/napl/sim/operation/add_any.py --hash "$(git hash-object src/napl/sim/operation/add_any.py)" \
    --changed yes \
    --gate "PASS (add_any + downstream sweep 46/46)" \
    --cpu "51.6->52.0 ms (1.00x)" \
    --gpu "22.6->17.3 ms (1.30x)" \
    --summary "in-place accumulator add_/clamp_ once shape matches; removes 2 allocs/timestep"
```

Record EVERY run, including a `--changed no` / ~1.00x outcome (no safe speedup found), so a
known-near-optimal kernel is visible in the log rather than silently re-attempted. Pass
`--date YYYY-MM-DD` only to backfill; it defaults to today. Confirm the recorded row to the user
as part of the verdict.

## Rules

Hard constraints on the Step 3 edit; a violation is a bug, not a speedup:

- **Numerical outputs MUST NOT change.** Bit-exactness is the target for integer-valued
  spike ops; only a legitimate float-reduction reorder may shift results within tolerance,
  and the test gate (Step 4) is what decides whether that is acceptable.
- Follow `CLAUDE.md` conventions: spike/non-spike dtypes (`self.stype`/`self.ntype`);
  **never use `>>`/`<<` on float tensors** (use `pow2_lshift`/`pow2_rshift` from
  `utils/utils.py`); keep `operation` imports at file top, since `operation` imports nothing
  from `module`, `metric`, or `algorithm`.
- **Preserve broadcast ability.** If the module sizes a running accumulator/buffer as a scalar
  (`torch.zeros(1)`) and grows it to the input shape on the first `forward()` via an out-of-place
  op, do NOT convert that to in-place: in-place cannot expand the destination and breaks the
  `(1,)` -> `(N,)` first-call broadcast. See the in-place broadcast failure mode for the safe
  shape-guarded pattern.
- If there is no safe speedup, change nothing and say so (report ~1.0x). Do not force a
  risky edit.

## References

- `scripts/bench.py`: `device_list()`, `sync(device)`, and `time_ms(fn, device, warmup,
  iters)`. The GPU-sync-correct timing primitives; import them rather than re-deriving the
  synchronize dance (the easiest thing to get wrong, and wrong = meaningless numbers).
- `scripts/record_opt.py`: append a uniform result row to `reports/napl-opt-sim-report.md`
  (Step 6). Creates the file with a header on first use, escapes table-breaking pipes, dedupes,
  and keeps the log sorted by kernel then date so re-runs of one kernel group together.
- `tests/<subpackage>/test_<kernel>.py`: the workload wiring and the correctness gate.
- `.claude/skills/napl-validate-unarysim/SKILL.md`: sibling skill with the same measurement
  rigor, applied to correctness-vs-UnarySim instead of speed. Reuse its test-wiring import
  pattern.
- `.claude/workflows/napl-port-unarysim.js`: the batch sweep whose Improve phase this skill is
  the interactive, single-kernel version of.
- `CLAUDE.md`: the project's testing rules (both devices, sync before timing, identical
  inputs) and the dtype/shift/import conventions the edit must respect.

## Failure modes

- **Identical inputs, before and after.** Generate inputs once under a fixed seed; the
  before and after timing runs must consume the same tensors, or the speedup (and any
  implicit correctness signal) is bogus. The #1 failure mode, same as napl-validate-unarysim.
- **Synchronize the GPU before stopping the clock.** `bench.time_ms` does this; if you time
  by hand, a missing `torch.mps.synchronize()`/`torch.cuda.synchronize()` measures launch
  latency, not compute, and will report a fantasty speedup.
- **MPS is a real device.** The committed tests' `cuda-or-cpu` idiom skips it; cover MPS
  explicitly via `device_list()`.
- **Reset between timed iterations.** Kernels are stateful (`timestep_cur`, internal
  registers). The timing thunk must `reset()` to a clean state each call, or later
  iterations start from dirty state and the numbers drift.
- **Behavior-preserving means the test passes unchanged.** Do not edit the test to make a
  faster-but-wrong kernel pass. The test is the gate, not a variable.
- **In-place ops do NOT type-promote; non-in-place ops do.** Switching `a.add(b)` /
  `a + b` to an in-place `a.add_(b)` / `a.clamp_(...)` to save an allocation is the most
  tempting hot-path win, but it silently changes the dtype contract. Non-in-place follows
  type promotion (`int8.add(float32)` -> a new `float32` tensor); in-place forces the
  result into the *destination's* dtype, which either **raises** (`int8.add_(float32)` ->
  `result type Float can't be cast to the desired output type Char`) or **truncates**. This
  bites in napl because `stype` (int8 spikes) and `ntype` (float32) mix throughout, so the
  promoted vs destination dtype genuinely differ. The trap: the test gate only catches it
  if the test exercises the regime where the dtypes differ. A bit-exact pass when operands
  happen to share a dtype (e.g. an `ntype`+`ntype` accumulator) does NOT prove the in-place
  form is safe where one operand is `stype`. Before any in-place swap, check the operand
  dtypes, and only do it when promotion would be a no-op (matched dtypes) AND the
  destination is not aliased/read elsewhere (`self.x.data` mutation can clobber a tensor a
  later line still needs). Honestly, an in-place rewrite usually buys little here (the cost
  is dominated by the per-timestep spike reduction, not accumulator-state allocation), so
  prefer leaving the non-in-place form unless profiling proves the allocation matters.
- **In-place ops break the scalar-accumulator broadcast.** This is the napl-specific in-place
  trap, separate from dtype promotion. Many modules init a running accumulator/buffer as a SCALAR
  (`torch.zeros(1)`) and rely on the **first** `forward()` to broadcast it up to the input shape via
  an *out-of-place* op: `self.acc.data = self.acc.add(delta)` turns `(1,)` + `(N,)` into a fresh
  `(N,)` tensor. Switching that to in-place (`self.acc.add_(delta)`) CANNOT grow the destination, so
  it raises on the first call (`output with shape [1] doesn't match the broadcast shape [N]`) or, worse,
  silently keeps the accumulator at `(1,)`. The optimization must preserve this broadcast: either keep
  the out-of-place form, or guard the in-place path on shapes already matching (the verified
  `add_any` pattern: `if self.accumulator.shape == acc_delta.shape: self.accumulator.add_(...) else:
  self.accumulator.data = self.accumulator.add(...)`, so the first timestep broadcasts out-of-place and
  steady state runs in place). The trap: a test that feeds a fixed-shape input from t=0 may never
  exercise the `(1,)` -> `(N,)` first-call expansion, so a bit-exact pass does NOT prove the broadcast
  still works. Check whether the module's `__init__`/`reset()` sizes its state as a scalar; if so,
  preserve the first-call out-of-place broadcast.
- **In-place breaks autograd.** The trainable binary-domain kernels (`*_hub`/`*_fxp`/
  `*_tlut`/`*_hard`, via `torch.autograd.Function` + STE) cannot have their forward
  intermediates mutated in place without corrupting the backward graph. In-place is a
  streaming-only consideration, and even there only under the dtype/aliasing rules
  above.
- **Stateful/import gotchas** carry over from CLAUDE.md: float bit-shift shims, the lazy
  `operation` imports, the decorrelated conv pad stream. Speeding up the hot path must not
  disturb these.
