---
name: napl-validate-unarysim
description: >-
  Validate a napl function/module against its reference implementation in UnarySim
  (github.com/diwu1990/UnarySim), the upstream simulator napl was ported from. Use this
  whenever the user wants to check that a napl op, kernel, metric, or layer matches
  UnarySim, confirm a port is faithful, verify fidelity against the upstream / "the
  original" / "the reference", reproduce UnarySim numbers in napl, or mentions UnarySim
  as the source of truth, even if they don't say "validate" outright (e.g. "does our
  mul_gaines match theirs?", "is correlation a faithful port?", "check linear against
  UnarySim"). It reads the UnarySim source (the local clone, GitHub as fallback), runs both
  implementations on identical inputs, and reports whether the numerical results agree. Sibling
  to napl-opt-sim (speed), napl-gen-rtl (RTL generation), and napl-validate-sim-rtl (sim vs RTL);
  the single-kernel counterpart of the napl-port-unarysim workflow's Validate phase. Every run
  leaves a durable row in reports/napl-validate-unarysim-report.md.
---

# Validate a napl function against its UnarySim reference

## Scope

napl is a from-scratch **port** of UnarySim's math into napl's conventions (see this repo's
`CLAUDE.md` and `references/experiment-plan.md`). UnarySim is the read-only source of truth; there is no
runtime dependency on it. "Faithful" means the two produce the **same numbers** on the same
inputs: **bit-exact** (`torch.equal`) for the integer-valued operands typical of unary/stochastic
computing, or within a stated tolerance / the stochastic-computing bound (~1/√N) when an
implementation legitimately reorders floating-point work. This skill compares one napl object to its
UnarySim original by running both, not by reading the code.

## Handoffs

- `napl-opt-sim`: sibling skill (speed) with the same measurement rigor applied to runtime.
- `napl-gen-rtl` / `napl-validate-sim-rtl`: sibling generate/validate pair for the RTL side.
- The `napl-port-unarysim` workflow's Validate phase is the batch sweep this skill is the
  single-kernel counterpart of.

## Persona

A skeptical validator who feeds identical inputs to both implementations, drives napl through the
exact committed test wiring rather than a re-implementation, and never calls a port faithful from
reading the code alone: run both, compare the numbers, explain any difference.

## Inputs

- The napl target to validate (e.g. `napl.sim.metric.correlation`, `napl.sim.operation.mul_gaines`,
  `napl.sim.module.linear`).
- Run everything through the project env: `conda run -n napl python ...` (a bare `python` is the
  wrong interpreter). Per this repo's testing rules, validate on **both CPU and GPU** (the GPU on
  Apple silicon is MPS: `torch.backends.mps.is_available()`).
- **Always compare against the local UnarySim clone at `/Users/diwu/Projects/UnarySim`** (this is the
  canonical reference). Import it as a package by putting its parent on the path:

  ```python
  import sys; sys.path.insert(0, '/Users/diwu/Projects')
  from UnarySim.kernel.linear import FSULinear      # or UnarySim.metric.metric, etc.
  ```

  This resolves the full dependency chain (RNG/BinGen/BSGen, FSUAdd, ...), so even the heavy kernels
  import directly - no standalone-file/`importlib` trick needed. (A `FutureWarning` about
  `torch.cuda.amp.autocast` from UnarySim is harmless.) If the clone is ever missing, fall back to the
  GitHub fetcher in Step 2.
- Heredocs piped through `conda run` swallow stdout. Write a `.py` file and run it, don't inline a
  `python - <<EOF`.

## Output contract

A short verdict with these fields: target, UnarySim class compared, inputs/regimes, the
**bit-exact result** (`torch.equal`) per device, the tolerance/RMSE vs the SC bound when not
bit-exact, and an explanation of any difference (deterministic-and-exact, stochastic-within-bound,
or a real divergence), plus a durable row in `reports/napl-validate-unarysim-report.md`.

## Workflow

### Step 0 - Always run the validation in a subagent

**Always delegate the validation to a subagent - never run the harness in the main thread.** A
validation is a self-contained, multi-step job (read the UnarySim source, write a harness, run it on
CPU and GPU, time it, interpret, record a log row) that produces a lot of transient output - source
dumps, harness iterations, stack traces, timing noise. Keeping that out of the main conversation is
the whole point: the orchestrator should end up with just the verdict and the logged row, not the
debugging trail.

So when this skill is invoked with a target, the main agent's job is **only** to dispatch and relay:

1. Spawn **one** `general-purpose` subagent (the `Agent`/`Task` tool) with a prompt like:

   > Validate napl `<target>` against its UnarySim reference. Follow the skill at
   > `.claude/skills/napl-validate-unarysim/SKILL.md` exactly: read it, then execute Steps 1-5 (map to
   > the UnarySim class, read the reference from the local clone `/Users/diwu/Projects/UnarySim`,
   > build a same-inputs harness, run on CPU and MPS, report the bit-exact result and any tolerance,
   > and append the row to `reports/napl-validate-unarysim-report.md` with the bundled `record_validation.py`). Return the
   > verdict table (bit-exact per device, runtimes) and the exact log row you appended.

2. When it returns, relay the verdict to the user and confirm the `reports/napl-validate-unarysim-report.md` row was
   written (read it back if unsure). If the subagent hit an API/device problem it couldn't resolve,
   re-dispatch with the extra context rather than finishing the work inline.

**If you ARE that dispatched subagent** (you were given a specific validation target and told to
follow this skill), ignore this section and just execute Steps 1-5 directly - do not spawn another
subagent. Run validations sequentially, one subagent each, so each log row is written cleanly.

### Step 1 - Identify the target and map it to a UnarySim class

Pin down the exact napl object (e.g. `napl.sim.metric.correlation`, `napl.sim.operation.mul_gaines`,
`napl.sim.module.linear`). Then find its UnarySim counterpart:

1. Check `references/mapping.md` (bundled) for the napl → UnarySim name + file mapping.
2. The napl class docstring and `references/experiment-plan.md`'s gap analysis also name the original.
3. If still unsure, search the local clone:
   `rg -l 'class <Name>' /Users/diwu/Projects/UnarySim` or
   `rg '^class ' /Users/diwu/Projects/UnarySim/kernel/<file>.py`.

UnarySim layout (same locally): `kernel/{mul,add,div,sqrt,signabs,relu,sigmoid,tanh,linear,conv,rnn,jkff,shiftreg,utils}.py`,
`metric/metric.py` (`Correlation`, `ProgError`, `Stability`), `stream/gen.py` (RNG/bitstream gen).

### Step 2 - Read the UnarySim reference (local clone)

The reference is the local clone at `/Users/diwu/Projects/UnarySim`; you `import` it (see Inputs),
you don't fetch it. Before writing the harness, **read the actual class** to get its API - `Read` or
`rg` the file directly, e.g.:

```bash
rg '^class ' /Users/diwu/Projects/UnarySim/kernel/linear.py      # list classes
sed -n '/class FSULinear/,/def forward/p' /Users/diwu/Projects/UnarySim/kernel/linear.py
```

Many UnarySim modules ship a matching test under `/Users/diwu/Projects/UnarySim/test/` that shows the
canonical wiring (how the input is encoded, what `forward` returns, how it's decoded) - read it; it is
the fastest way to get the harness right.

The API varies by era of UnarySim, so check the read source for:
- the constructor signature (older code takes positional args / `mode`; newer code takes `hwcfg`/`swcfg`
  dicts),
- the per-timestep entry point (metrics split update vs. report: `Monitor()` accumulates, `forward()`
  reports; kernels usually just have `forward()`),
- how it binarizes / what dtype it works in (often `torch.float`).

### Step 3 - Build a side-by-side harness on identical inputs

The cardinal rule: **generate the inputs once and feed the same tensors to both implementations.**
Regenerating random inputs per run (or per implementation) silently compares different data and is
the most common way to get a bogus "they don't match" (or a bogus match). This bit me twice; guard
against it.

**Reuse the committed test wiring for the napl side - do not re-write it.** Most ops/kernels have a
`tests/.../test_<op>.py` that defines a `napl_<op>` module class (encode(s) -> op -> decoder, run by
`@napl_sim_timesteps`). Import that class and use it as the napl half of the harness, so the
validation exercises napl through *exactly* the code the test suite runs - it can't drift from how the
op is actually wired, which is what makes the validation trackable. A re-implemented harness can pass
while the real wiring is broken (or vice versa); reusing the test class closes that gap.

```python
import importlib.util
spec = importlib.util.spec_from_file_location('t', 'tests/module/test_linear.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)   # __main__ block does NOT run
NaplWiring = t.napl_linear         # the committed wiring class
```

The test wiring encodes its spikes internally, so to feed UnarySim *the same* stream you don't extract
them - you **mirror the test's encoder configs**. napl encoders are deterministic: two encoders with
the same config fed the same input value emit an identical spike stream (verified). So generate the
input value once, hand it to both the test-wiring instance and a set of mirror encoders built from the
test's exact `codec_config`s (read them from the test file), and both sides consume identical inputs.

If the matching test **inlines** its wiring instead of using a `napl_<op>` class (e.g.
`test_correlation.py`, `test_stability.py`, the binary-domain `*_hub`/`*_fxp`/`*_tlut`/`*_hard` tests),
mirror the test's exact configs and call sequence in the harness, and note in the verdict that no
reusable wrapper existed. (Refactoring such a test to expose a `napl_<op>` class is a good follow-up,
since it makes future validations trackable too.)

napl has two execution paradigms (see `CLAUDE.md`); the harness differs:

**A. Streaming op or metric** (per-timestep `forward()`; every call auto-ticks `timestep_cur`): drive both over N timesteps
with the *same* spike stream each step, then compare the decoded value / metric report. The example
below is `correlation`, whose test inlines its wiring, so it builds the napl side directly; for an op
that has a `napl_<op>` test class, **import and drive that class** (per the reuse rule above) instead
of constructing the op inline, and mirror its encoder configs for the UnarySim side.

```python
import sys, torch
sys.path.insert(0, '/Users/diwu/Projects')          # local UnarySim clone's parent
from UnarySim.metric.metric import Correlation
from napl.sim.operation import encode
from napl.sim.metric import correlation
from napl.utils import gen_rand_tensor

T = 256
torch.manual_seed(0)
val = gen_rand_tensor('bipolar', shape=(1000,), width=8)          # generated ONCE
enc = encode({'polarity': 'bipolar', 'timestep': T, 'generator': 'sobol', 'dim': 1})

nap = correlation()
ref = Correlation()
# UnarySim metric accumulators are scalar (1,) and updated with in-place add_, which cannot
# broadcast to a vector input; pre-size them to the input length so a fair comparison runs.
for nm in ('paired_00_d','paired_01_c','paired_10_b','paired_11_a','in_1_d'):
    if hasattr(ref, nm): getattr(ref, nm).data = torch.zeros(1000)

enc.reset()
for _ in range(T):
    s = enc(val)
    nap(s, s)                       # same spikes to both
    ref.Monitor(s.type(torch.float), s.type(torch.float))
out_nap = nap.analyze()[0]          # napl stats method, returns (value, argmax)
out_ref = ref.forward()             # UnarySim's report method
print('equal:', torch.equal(out_nap, out_ref), 'max|diff|:', (out_nap-out_ref).abs().max().item())
```

**B. Single-shot binary-domain kernel** (`*_hub`/`*_fxp`/`*_tlut`/`*_hard`, whole-tensor `forward()`):
build both with the **same weights** (pass napl's weight tensor into the UnarySim module, or copy
UnarySim's into napl) and the same input, then compare outputs. These are also checked against the
float reference (`nn.Linear`/`nn.Conv2d`) within the quantization bound, so a three-way check
(napl vs UnarySim vs `nn.*`) is often the most informative.

Cover the regimes that matter for the op, not just one random case: e.g. both polarities
(unipolar/bipolar), the known-answer corners (a stream vs itself, vs its complement, vs an
independent stream), and an edge input (all-zero) when relevant.

### Step 4 - Compare and interpret

**Always compute and report the bit-exact result** (`torch.equal(out_nap, out_ref)` and
`max|diff|`) for every regime, even when SC-bound agreement is the acceptable outcome. The
bit-exact flag is the single most informative line of the verdict: it distinguishes a
**deterministic** port (a fixed function of the same input spikes - metrics, gate ops - which
should be bit-exact) from a **stochastic** one (a kernel that encodes an operand with its own
internal RNG - `linear`, `conv` weight streams - which will not be bit-exact and is only
expected to agree within ~1/√N). Print `bit-exact=<bool>` next to the tolerance/RMSE so the reader
sees both; never report a tolerance pass while hiding that the outputs are (or aren't) identical.

- **Bit-exact** (`torch.equal`, `max|diff| == 0`) is the expected outcome whenever the result is a
  deterministic function of the same input spikes (no internal RNG): metrics, gate primitives, and
  any kernel fed identical operand streams. The values are 0/1 spikes and small integer partial
  sums, exact in float, so the bar is exact equality, not "close".
- **Tolerance** (`bit-exact=False` but still faithful): two cases. (1) An implementation legitimately
  reorders float work (e.g. napl replaces an elementwise spike-product reduction with a `matmul`), so
  bit-exactness may not survive across devices for general floats - tie the tolerance to the
  exact-integer argument if it still holds. (2) A streaming kernel encodes an operand with its own
  RNG, so the two never match bit-for-bit; report `bit-exact=False` and check that both track the
  analytic / `nn.*` reference AND each other within a few times the SC bound (~1/√N). In both cases
  state the bit-exact flag *and* the tolerance, not just the tolerance.
- **A real divergence** (results differ beyond rounding) is a finding: locate it (compare the
  pre-decode partial sums, not just the final value), say which side is right relative to the math /
  the `nn.*` reference, and report it rather than hand-waving.
- **Benign reformulations are not divergences.** napl frequently stores different intermediate state
  than UnarySim for speed or clarity (e.g. correlation keeps sufficient statistics and derives the
  bins at report; accuracy hardcodes a float accumulator). Confirm the *outputs* match and note the
  reformulation; don't flag it as a mismatch.

Deliver a short verdict with these fields: target, UnarySim class compared, inputs/regimes, the
**bit-exact result** (`torch.equal`) per device, the tolerance/RMSE vs the SC bound when not
bit-exact, and an explanation of any difference (deterministic-and-exact, stochastic-within-bound,
or a real divergence).

### Step 5 - Record the validation (always)

Every validation must leave a durable row in **`reports/napl-validate-unarysim-report.md`** so the port's fidelity is
tracked over time. Before recording, measure the napl module's **CPU and GPU (MPS) runtime** for the
validated workload - the same per-timestep loop you already built, timed with `torch.mps.synchronize()`
before stopping the GPU clock (an untimed-sync GPU number is meaningless). Then append the row with the
bundled script (it creates the file with a header on first use and keeps every row uniform):

```bash
conda run -n napl python .claude/skills/napl-validate-unarysim/scripts/record_validation.py \
    --module module.linear --ref FSULinear \
    --bitexact "no (internal weight RNG)" \
    --agreement "RMSE 0.0014 vs SC bound 0.031, CPU+MPS" \
    --cpu "93.2 ms" --gpu "53.4 ms" \
    --regimes "bipolar, unipolar; B=64 in=512 out=256 T=256"
```

The script keeps the log **sorted by module name** (then by date for re-validations) on every write,
and dedupes identical rows - so re-validating a module groups its history together rather than
appending chronologically at the bottom. Pass `--date YYYY-MM-DD` only to backfill; it defaults to
today. Use `--gpu "n/a (unsupported on MPS)"` if a device genuinely can't run the op. Confirm the
recorded row to the user as part of the verdict.

## Rules

- **Identical inputs or it's meaningless.** Generate the inputs once and feed the same tensors to
  both implementations (Step 3). The #1 failure mode.
- **Reuse the committed test wiring** for the napl side (import the `napl_<op>` class); do not
  re-implement it, so the validation exercises exactly the code the test suite runs.
- **Run both and compare; never assert a match from reading the code alone.**
- **Always compute and report the bit-exact result** (`torch.equal`, `max|diff|`) for every regime,
  alongside any tolerance/RMSE (Step 4).
- Validate on **both CPU and GPU (MPS)**, and synchronize the GPU before stopping any timing clock.

## References

- Local UnarySim clone: **`/Users/diwu/Projects/UnarySim`** (canonical reference; import via parent on
  `sys.path`). Its `test/` dir has runnable examples of the canonical wiring per kernel.
- `references/mapping.md` - napl → UnarySim name + file mapping, and the UnarySim repo layout.
- `scripts/record_validation.py` - append a uniform validation row to `reports/napl-validate-unarysim-report.md` (Step 5).
- `reports/napl-validate-unarysim-report.md` (in the napl repo) - the running validation log this skill maintains.
- `scripts/fetch_unarysim.py` - **fallback only** (use when the clone is missing): fetch a UnarySim
  source file from GitHub by repo path; lists its classes.

## Failure modes

- **Identical inputs or it's meaningless** - see Step 3. The #1 failure mode.
- **UnarySim scalar accumulators don't broadcast** - metric/stateful modules init buffers as `(1,)`
  and update with in-place `add_`, which errors on vector inputs (`shape [1] doesn't match broadcast
  shape [N]`). Pre-size the buffers to the input length; this is napl being more robust, not an
  algorithmic difference.
- **Binarization equivalences** - UnarySim's `1 - eq(x,0)` is napl's `ne(x,0)`; same function. Don't
  mistake a rewrite for a divergence.
- **Heavy kernels import fine from the clone** - `FSULinear`/`FSUConv2d`/etc. pull in `RNG`, `FSUAdd`,
  `@autocast()` from across the package, so they can't be loaded as a single standalone file; importing
  from the installed-on-path clone resolves the whole chain. (This is why the local clone is canonical;
  the GitHub fetcher only worked for torch-only files like `metric/metric.py`.)
- **Method-name mapping** - UnarySim metrics: `Monitor()` = accumulate, `forward()` = report. napl:
  `forward()` = accumulate, `analyze()` = stats (lazy value properties in between). Kernels on both
  sides use `forward()`.
- **API era** - check the clone's constructor: older UnarySim is positional/`mode=`, newer takes
  `hwcfg`/`swcfg` dicts. Match whatever the local source actually defines.
