---
name: napl-gen-sim
description: >-
  Port a UnarySim class into napl: find UnarySim modules/kernels/metrics that napl does not yet
  have, and reimplement them in napl's conventions (napl_base, config validation, the streaming
  per-timestep paradigm, hw_params), with a test and a faithfulness check against the original.
  Use whenever the user wants to port from UnarySim to napl, "add the missing UnarySim kernels",
  "implement <UnarySim class> in napl", "what's in UnarySim that napl lacks", "scan for unported
  classes", "bring FSULinearPC over", or otherwise fill napl's gaps from the upstream simulator,
  even without the word "port". It is the generate side of the port (UnarySim -> napl); the
  validate side is napl-validate-unarysim, which it invokes to confirm each port is faithful.
  Sibling to napl-gen-rtl (lowers napl to Verilog) and napl-opt-sim (speeds napl up); the
  per-class porting counterpart of the napl-port-unarysim workflow. Every run leaves a durable
  row in reports/napl-gen-sim-report.md.
---

# Port a UnarySim class into napl

## Scope

napl is a from-scratch port of UnarySim's math into napl's conventions, but the port is not
complete and UnarySim keeps growing. This skill closes the gap: it finds UnarySim classes with no
napl counterpart and reimplements them **in napl's style** (not a line-for-line copy), gives each a
test, and confirms it reproduces the UnarySim original. It is the producer counterpart of
napl-validate-unarysim (which validates an existing port) and the same generate-then-validate shape
as napl-gen-rtl/napl-validate-sim-rtl.

## Handoffs

- `napl-validate-unarysim`: invoked in Step 5 to confirm each port is faithful to the UnarySim
  original.
- `coding-discipline`: invoked first in Step 3 as the coding overlay for the reimplementation.
- `napl-gen-rtl` / `napl-validate-sim-rtl`: sibling generate/validate pair for the RTL side.
- The `napl-port-unarysim` workflow invokes this skill per gap class with the usual OVERRIDES:
  do NOT run the Step 6 recorder yourself (the workflow records centrally via a serialized
  recorder), and return the structured result instead. Standalone, do Step 6 normally.

## Persona

A careful porter who reimplements the math in napl's idiom rather than transcribing UnarySim line
by line, checks the mapping before declaring a gap, and never calls a port done until it both
passes its own test and agrees with the upstream original.

## Inputs

- The UnarySim class name(s) to port, or a request to scan for the whole gap.
- Run through the project env: `conda run -n napl python ...` (a bare `python` is the wrong
  interpreter). Heredocs piped through `conda run` swallow stdout, so write a `.py` file.
- The UnarySim source is the **local clone at `/Users/diwu/Projects/UnarySim`** (read-only reference);
  import it via its parent on the path (`sys.path.insert(0, '/Users/diwu/Projects')`).

## Output contract

- A new napl class wired into its subpackage, a `test_<name>.py`, an updated mapping.md
  correspondence, a recorded faithfulness verdict, and a durable row in
  `reports/napl-gen-sim-report.md` (including `skipped`/`failed` runs).

## Workflow

### Step 0 - Always run the port in a subagent

**Always delegate to a subagent; never port in the main thread.** Reading the UnarySim source,
reimplementing it, writing a test, and validating it is a multi-step job with heavy transient output.
When invoked, the main agent dispatches and relays:

1. To **scan and port the whole gap**: spawn one `general-purpose` subagent to identify every
   unported UnarySim class (Step 1) and return the list; then spawn **one subagent per class** to
   port it (Steps 2-6), one class at a time so each report row and `__init__`/mapping edit lands
   cleanly (concurrent agents would race on shared files like `module/__init__.py`).
2. To **port a named class**: spawn one subagent for that class with Steps 2-6.

   > Port the UnarySim class `<ClassName>` into napl. Follow the skill at
   > `.claude/skills/napl-gen-sim/SKILL.md` exactly: read it, then execute Steps 2-6 (read the
   > UnarySim source, reimplement in napl conventions, wire it into its subpackage, write
   > `test_<name>.py`, validate faithful against UnarySim via napl-validate-unarysim, record the row).
   > Return the napl class created, the test, and the validation verdict.

Relay each port's verdict. **If you ARE the dispatched subagent**, execute the Steps directly (do not
re-dispatch). Port one class per subagent.

### Step 1 - Scan for the gap (which UnarySim classes napl lacks)

Start from the existing record. `.claude/skills/napl-validate-unarysim/references/experiment-plan.md`
is the project's porting gap analysis ("port all UnarySim metrics & kernels into napl") with a
`## Gap analysis (UnarySim main -> napl)` section, a `To port:` list, and the test/convention patterns
each item must follow; read it first to see what was already planned, ported, and deferred. Then list
UnarySim's classes (`kernel/`, `metric/`, `stream/`) and cross-reference napl to find which have **no
napl counterpart**. The authoritative name correspondence is the bundled-next-door
`.claude/skills/napl-validate-unarysim/references/mapping.md` (napl <-> UnarySim names + files), since
names differ (UnarySim `FSUMul` is napl `mul_csg`, not a missing class). A class is a real gap only if
the experiment-plan, mapping.md, and a search of `src/napl/` all show no napl equivalent.

```bash
grep -rhoE '^class +[A-Za-z0-9_]+' /Users/diwu/Projects/UnarySim/{kernel,metric,stream} | sed 's/class //' | sort -u
```

**Skip non-ports:** backward-only `torch.autograd.Function` helpers (`*Function`, the STE shims behind
HUB/FXP/TLUT layers) are not standalone napl modules; the `RNG`/`RawScale`/`BinGen`/`BSGen` stream
generators correspond to napl's `encoder`/`decoder`/`gen_num_seq` codec, not new classes. The genuine
gaps today are the parallel-counter variants (`FSULinearPC`, `FSUConv2dPC`). Record a skip with its
reason rather than forcing an unnatural port.

### Step 2 - Read the UnarySim implementation as the spec

Read the UnarySim class from the local clone (constructor, `forward`, any helper state). Note its
**paradigm**: a per-timestep streaming kernel (consumes one spike per call, accumulates over
timesteps) maps to napl's streaming form; a whole-tensor binary-domain layer (HUB/FXP/TLUT, trainable
via `autograd.Function`) maps to napl's single-shot form. Note its polarity handling, its RNG/Sobol
use, and how it holds state. Its matching `test/` file in the clone shows the canonical wiring.

### Step 3 - Reimplement in napl conventions (not a line-for-line copy)

FIRST invoke the **`coding-discipline`** skill and follow it (think before coding, simplicity first,
surgical changes, verifiable criterion). Then write the napl class in the correct subpackage
(`operation/` for gate-level primitives, `module/` for neural layers, `metric/` for metrics),
following napl's style exactly (see `CLAUDE.md` and the nearest existing sibling, e.g. port
`FSULinearPC` in `module/linear_pc.py`):

- **`napl_base` subclass** taking a single `config` dict; call
  `super().__init__(config, key_list, polarity_required)` to validate required keys. Use napl naming
  (lowercase, descriptive: `mul_csg`, `linear`, not `FSUMul`); pick a name consistent with the
  existing family (a PC variant of `linear` -> `linear_pc`).
- **Dtypes:** spike tensors are `self.stype`, non-spike `self.ntype`. NEVER use `>>`/`<<` on float
  tensors (use `pow2_lshift`/`pow2_rshift` from `utils/utils.py`).
- **Streaming kernels:** describe ONE timestep in `forward()`; every call auto-ticks
  `timestep_cur` (no manual `self.tick()`), `reset()` clears it; the round-trip is encoder -> op ->
  decoder run by `@napl_sim_timesteps`. **Binary-domain layers:** whole-tensor `forward()` (they set
  `streaming = False`, so `timestep_cur` stays 0), trainable via `torch.autograd.Function` + STE.
- **Lazy readouts:** apply this pattern only to spike-domain endpoints, meaning sinks whose readout
  no downstream spike operation consumes. The endpoints are `decoder` and the five metric classes
  `accuracy`, `correlation`, `stability`, `stability_norm`, and `stability_flux`. Keep their
  `forward()` methods accumulate-only and expose derived readouts through on-demand properties.
  Sources such as `encoder` and in-stream spike operations must keep returning their per-timestep
  spikes. See `src/napl/sim/metric/accuracy.py:50-75` and
  `src/napl/sim/module/decoder.py:30-58`.
- **Adapt, do not transcribe, UnarySim's state idioms.** UnarySim pre-sizes scalar buffers and updates
  in place; napl's idiom is a scalar (`torch.zeros(1)`) accumulator that **broadcasts up to the input
  shape on the first `forward()` via an out-of-place op** (`self.acc.data = self.acc.add(delta)`).
  Follow napl's broadcasting idiom so the class works on vector inputs without pre-sizing.
- **hw_params:** if it is an `operation` with a gate-level mapping, set
  `self.hw = hw_params(pp_delay=...)` (`from napl.sim.base import napl_base, hw_params`).
- **Imports:** import `operation` primitives **lazily inside `__init__`** (the `module`<->`operation`
  import cycle). Wire the new class into its subpackage `__init__.py` (mind import order).

### Step 4 - Write the test and update the mapping

Add `tests/<subpackage>/test_<name>.py` by copying the matching skeleton:
`tests/template_streaming_kernel.py` for per-timestep kernels, or
`tests/template_single_shot_trainable.py` for trainable binary-domain kernels. Fill in every TODO
(see `tests/operation/test_mul_and.py` for streaming wiring), run on **every device** (CPU +
MPS/CUDA), and check the correctness and performance gates in `RULE_SIM.md`, with an
`if __name__ == '__main__'` entry point. Add the new correspondence to
`.claude/skills/napl-validate-unarysim/references/mapping.md` so future validations find it.

### Step 5 - Validate the port is faithful

Run the new test (`conda run -n napl pytest tests/<subpackage>/test_<name>.py`) on every device. Then
confirm faithfulness against the UnarySim original: invoke the **`napl-validate-unarysim`** skill on
the new class (execute its Steps directly; feed identical inputs to both, report bit-exact or
within-bound agreement). A port is not done until it both passes its own test and agrees with
UnarySim. If it cannot be made faithful, leave it as `failed` and explain the divergence.

### Step 6 - Record the run (always)

Append a durable row to **`reports/napl-gen-sim-report.md`** with the bundled recorder (header on
first use, dedupe, sorted by napl class). Record every run, including `skipped`/`failed`:

```bash
conda run -n napl python .claude/skills/napl-gen-sim/scripts/record_gen_sim.py \
    --napl module.linear_pc --unarysim FSULinearPC \
    --status ported --test tests/module/test_linear_pc.py \
    --validated "bit-exact vs FSULinearPC (CPU+MPS)" \
    --notes "parallel-counter variant of linear; streaming kernel"
```

Deliver the verdict: napl class created, UnarySim source, subpackage/paradigm, test, validation
result, and (if skipped/failed) why. Confirm the recorded row.

## Rules

- **Port the math, not the names.** napl names are lowercase and descriptive (`mul_csg`, not
  `FSUMul`); match the existing family's naming so the new class reads like its siblings.
- **Different names are not missing classes.** Check mapping.md before declaring a gap; most UnarySim
  classes are already ported under a napl name.
- **Streaming vs single-shot is the first decision.** A per-timestep spike kernel and a whole-tensor
  binary-domain layer are structured completely differently in napl (per-timestep `forward()` +
  `@napl_sim_timesteps` vs autograd `Function`); get the paradigm right before writing code.
- **Adopt napl's broadcast idiom.** Do not transcribe UnarySim's pre-sized in-place buffers; use the
  scalar accumulator that broadcasts to input shape on the first call (see Step 3), so the port runs
  on vector inputs.
- **Float-shift / lazy-import / pow2 conventions** from CLAUDE.md apply to the ported code too.

## References

- `scripts/record_gen_sim.py` - append a uniform row to `reports/napl-gen-sim-report.md` (Step 6).
- `.claude/skills/napl-validate-unarysim/references/experiment-plan.md` - the project's port gap
  analysis: what to port, the test/convention patterns, and what is already done vs deferred. Read it
  first in Step 1.
- `.claude/skills/napl-validate-unarysim/references/mapping.md` - the authoritative napl <-> UnarySim
  correspondence and the gap list; read first, and update after a port.
- `.claude/skills/napl-validate-unarysim/SKILL.md` - the validate counterpart this skill invokes in
  Step 5 to confirm faithfulness.
- `CLAUDE.md` - napl's conventions (napl_base, config, streaming vs single-shot, dtypes, hw_params,
  the float-shift/lazy-import gotchas) the port must follow.
- `tests/template_streaming_kernel.py` - the copy-paste skeleton for per-timestep kernels.
- `tests/template_single_shot_trainable.py` - the copy-paste skeleton for trainable binary-domain kernels.
- `tests/operation/test_mul_and.py` - the canonical worked example for a streaming operation.
- `/Users/diwu/Projects/UnarySim` - the local clone (read-only source); its `test/` dir shows the
  canonical wiring per class.
- `.claude/skills/napl-gen-rtl/SKILL.md` - sibling generator (napl -> Verilog), same generate/validate
  shape.

## Failure modes

- **A port is not done until validated.** Passing a self-written test is necessary but not sufficient;
  it must also agree with the UnarySim original (Step 5). If it cannot be made faithful, record it as
  `failed` with the divergence explained, not `ported`.
- **Skip honestly.** Backward-only autograd helpers and the RNG/bitstream generators are not new napl
  classes; record a skip with the reason rather than inventing a class.
