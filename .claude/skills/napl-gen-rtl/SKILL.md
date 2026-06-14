---
name: napl-gen-rtl
description: >-
  Generate the synthesizable Verilog RTL for a napl operation, verified against its Python
  functional model by golden-vector co-simulation. Use whenever the user wants to create,
  write, emit, or lower a napl op to Verilog/RTL/hardware ("generate RTL for mul_and",
  "write the Verilog for add_any", "implement shiftreg in hardware", "lower sqrt_emit to
  RTL", "make the hardware for this op"), even without the word "RTL". It emits one module
  per polarity variant plus a testbench and a golden-vector generator under
  src/napl/implementation/<op>/, makes `make test OP=<op>` PASS bit-exactly, and writes the
  RTL pipeline delay back to the class's self.hw.pp_delay. Operation subpackage only: napl
  modules/metrics/layers are not gate-level circuits. Sibling to napl-validate-sim-rtl
  (which validates an existing RTL against the test's inputs) and to the napl-port-unarysim
  workflow's RTL phase (whose generation this skill is the single-op, interactive version
  of). Every run leaves a durable row in reports/napl-gen-rtl-report.md.
---

# Generate Verilog RTL for a napl operation

## Why this exists

napl lowers each gate-level operation to a synthesizable Verilog counterpart under
`src/napl/implementation/<op>/`. This skill produces that RTL for one operation: the module(s),
a self-checking testbench, and a generator that emits golden vectors **from the napl Python
model** so the hardware is checked against the actual simulator, not a hand-written truth table.
It is the single-op, interactive version of the napl-port-unarysim workflow's RTL phase, and the
producer side of napl-validate-sim-rtl (which validates an already-generated RTL against the test's
inputs). The deliverable is a working op directory where `make test OP=<op>` PASSes bit-exactly,
with the RTL pipeline delay recorded back into the class's hardware contract.

## Execution model: always run the generation in a subagent

**Always delegate to one subagent; never run it in the main thread.** Writing RTL + testbench +
generator and iterating until `make test` passes emits a lot of transient output (Verilog drafts,
iverilog/Verilator lint, vvp mismatch traces). Keep it out of the main conversation. When invoked
with an operation target, the main agent only dispatches and relays:

1. Spawn **one** `general-purpose` subagent (`Agent`/`Task` tool) with a prompt like:

   > Generate Verilog RTL for the napl operation `<op>`. Follow the skill at
   > `.claude/skills/napl-gen-rtl/SKILL.md` exactly: read it, then execute Steps 1-6 (resolve
   > the op; design the RTL from the Python model; emit rtl/tb/gen; make `make test OP=<op>` PASS;
   > write pp_delay back to self.hw; record the row). Return the result table and the exact
   > `make test` output.

2. Relay the result (status, make test, pp_delay, module names). If the op has no sound gate-level
   mapping, surface that it was skipped rather than forcing an unsound design.

**If you ARE that dispatched subagent**, ignore this section and execute Steps 1-6 directly.
Generate one operation per subagent.

## Environment

- Run through the project env: `conda run -n napl python ...` and
  `conda run -n napl make test OP=<op>` (from `src/napl/implementation/`). A bare `python` is the
  wrong interpreter. Heredocs piped through `conda run` swallow stdout, so write a `.py` file.
- The co-sim needs the **Icarus Verilog** toolchain (`iverilog`/`vvp`) on PATH. If missing, say so
  rather than claiming a result.
- Editing a `.v` file auto-injects the global **Verilog rules** via the `verilog-rules.sh` hook
  (Verilog-2001; clock `i_clk`; active-low `i_rst_n`; `i_`/`o_` port prefixes; `_n` for active-low;
  one module per file with filename == module; Verilator-lint-clean). Follow them.

## Workflow

### Step 1 - Resolve the operation

Pin down the napl `operation` class (e.g. `mul_and`, `shiftreg`, `add_any`), its source under
`src/napl/operation/`, and its `tests/operation/test_<op>.py`. The op directory and `make test`
OP name is the class name: `src/napl/implementation/<op>/`. **Operation subpackage only** - napl
`module`/`metric`/`algorithm` classes are whole-tensor or statistical, not per-timestep gate
circuits, so they have no RTL; report that and stop if asked for one.

### Step 2 - Read the Python model as the spec

The Python `forward()` is the contract. Read it (and `__init__`/`reset()`) for: the per-timestep
logic and its polarity branches (e.g. `mul_and` is AND unipolar / XNOR bipolar); what state it
holds; what `reset()` initializes, **including non-zero reset state** (`shiftreg` reloads
`reg[i] = i % 2`, not zeros); and the input/output spike shape. If a matching module exists in
UnarySim's RTL you may refer to it for structure, but the napl model is the source of truth.

### Step 3 - Emit the RTL, testbench, and generator

FIRST invoke the **`coding-discipline`** skill (Skill tool) and follow it across both the Verilog and
the Python generator/testbench: think before coding (sketch the datapath and register stages from the
model before writing a line), simplicity first (the smallest gate-level circuit that reproduces
`forward()`, no speculative parameters or modes), surgical changes (touch only this op's folder), and
goal-driven execution with a verifiable success criterion (`make test OP=<op>` PASSes bit-exactly).

Lay the op out self-contained under `src/napl/implementation/<op>/{rtl,tb,gen,vec,build}/`,
following `src/napl/implementation/README.md` and the auto-injected Verilog rules:

- **`rtl/*.v`** - one synthesizable module per concrete variant. **Module naming:** the module name
  (and its filename) is *exactly* the op name (which may contain underscores), optionally followed by
  a SINGLE polarity postfix `_unipolar` or `_bipolar` when the op has distinct polarity variants. No
  other postfix (no mode/config/version/width tags). No polarity split -> bare op name.
- **Timing mapping:** one Python `forward()` timestep == one `posedge i_clk`. Active-low `i_rst_n`
  maps to the Python `reset()`, and must bring every register to the EXACT post-`reset()` state of
  the model (not always zero - reload whatever `reset()` sets).
- **`gen/gen_<op>.py`** - emits `vec/<op>.vec` with golden vectors **from the napl Python model**
  (`from napl.operation import <op>`), never a hand-derived truth table. Drive the model per timestep
  from `model.reset()` at t=0 and record per-cycle (inputs, output), one polarity column per variant
  (see `mul_and`'s `in_0 in_1 out_unipolar out_bipolar`). The bit-serial RTL has one 1-bit datapath,
  so drive scalar streams.
- **`tb/<op>_tb.v`** - self-checking testbench: replays the vectors cycle by cycle (pulse `i_rst_n`
  low before driving so the co-sim starts from t=0), compares the RTL output to the expected column,
  and prints `PASS ...` ONLY on a full bit-exact match (the Makefile greps for `^PASS`).

Copy the nearest live pattern: `implementation/mul_and/` (combinational) or `implementation/shiftreg/`
(stateful, non-zero reset). Do NOT edit the shared `Makefile` - it is already generic via `OP=`.

### Step 4 - Verify with the co-simulation

From `src/napl/implementation/`, run `conda run -n napl make test OP=<op>`: it runs your generator,
compiles `rtl/*.v` + the testbench with `iverilog -g2012 -Wall`, simulates with `vvp`, and PASSes
only on a full match. Iterate the RTL until it PASSes bit-exactly. **Bit-exact is the bar** - spikes
are 0/1 and the RTL is a faithful gate-level model, so it matches or there is a real bug.

If the class is **not a per-timestep streaming circuit** (no natural gate-level mapping), do not
force an unsound design: set status `skipped` (pp_delay blank) and record why.

### Step 5 - Determine and write pp_delay

Compute `pp_delay`: the RTL input-to-output latency in `i_clk` cycles (0 if purely combinational,
1 per register stage). When input->output paths or multiple outputs have DIFFERING latencies, report
the **minimum** across them (the napl-port-unarysim convention). Write it back to the class's hardware
contract: `self.hw = hw_params(pp_delay=<n>)` in `__init__` after `super().__init__(...)` (update the
`pp_delay=` arg if a `self.hw = hw_params(...)` line exists; replace a legacy `self.delay = n`; add it
otherwise). Ensure the file imports it: `from napl.base import napl_base, hw_params`. See
`operation/mul.py` and `operation/shiftreg.py` for the idiom. This field must match the RTL for sim/HW
timing to agree.

### Step 6 - Record the run (always)

Append a durable row to **`reports/napl-gen-rtl-report.md`** with the bundled recorder (header on
first use, dedupe, sorted by class). Record EVERY run, including `skipped` and `failed`, so a
by-design skip is visible rather than silently absent:

```bash
conda run -n napl python .claude/skills/napl-gen-rtl/scripts/record_gen_rtl.py \
    --class shiftreg --rtl shiftreg \
    --status verified --make-test PASS \
    --pp-delay 4 --polarities "none (bare op)" \
    --notes "depth-4 circular FIFO; non-zero i%2 reset; gen + tb from the Python model"
```

Deliver the verdict: class, RTL module(s), status, `make test` result, pp_delay, polarities, and
(if skipped/failed) why. Confirm the recorded row.

## When the napl-port-unarysim workflow calls this skill

The workflow's RTL phase invokes this skill per operation in a fan-out, with two OVERRIDES (so its
central ledgers stay the source of truth and shared source files are not edited concurrently):

- **Do NOT write `self.hw.pp_delay` yourself** (skip Step 5's file edit). Operation classes share
  files (e.g. `compare.py`), so the workflow's Finalize phase applies all pp_delay edits serially.
  RETURN the pp_delay instead.
- **Do NOT run the Step 6 recorder** (`reports/napl-gen-rtl-report.md`). The workflow records the
  outcome centrally to `napl-port-unarysim/implementation.md`.

Standalone (a user invoking this skill directly), do both Step 5 and Step 6 normally.

## Bundled resources

- `scripts/record_gen_rtl.py` - append a uniform row to `reports/napl-gen-rtl-report.md` (Step 6).
  Creates the file with a header on first use, escapes table-breaking pipes, dedupes, and sorts by
  class name.

## Gotchas (learned the hard way)

- **Vectors come from the model, never a truth table.** The expected column is whatever the napl
  `forward()` emits. A hand-derived expectation validates the RTL against your arithmetic, not napl.
- **Reset to the model's actual state, from t=0.** `i_rst_n` low must reproduce the EXACT post-
  `reset()` registers, which are not always zero (`shiftreg` reloads `i % 2`). Drive the model from
  `reset()` and pulse `i_rst_n` low before the stream; a wrong reset value fails on the opening cycles.
- **One module per polarity, exact naming.** Distinct unipolar/bipolar variants are separate modules
  with a single `_unipolar`/`_bipolar` postfix; never a mode/width/version tag. Filename == module.
- **pp_delay is min across paths, written to `self.hw`, not `self.delay`.** Use the `hw_params`
  contract and add the `hw_params` import; the legacy scalar `self.delay` is superseded.
- **Bit-serial scalar datapath.** One 1-bit datapath per (op, variant); the generator drives scalar
  streams, not the test's wide tensor. Vectorization across lanes is a higher-level block's job.
- **Skip honestly.** If an op has no gate-level mapping, status `skipped` beats an unsound design.
- **iverilog must be on PATH.** No toolchain, no co-sim; say so rather than reporting an unrun result.
- **Do not edit the shared Makefile.** It is already generic via `OP=`.

## References

- `src/napl/implementation/README.md` - the RTL spec: layout, naming, combinational vs clocked, the
  `i_clk`/`i_rst_n` and reset-state contract.
- `src/napl/implementation/mul_and/` - canonical combinational example (rtl, tb, gen).
- `src/napl/implementation/shiftreg/` - canonical stateful example: per-cycle stream, non-zero reset.
- `src/napl/base/base.py` - `hw_params` (the `pp_delay` contract); `operation/mul.py`,
  `operation/shiftreg.py` show the `self.hw = hw_params(pp_delay=...)` idiom.
- `.claude/skills/napl-validate-sim-rtl/SKILL.md` - sibling: validates a generated RTL against the
  test's inputs (the verification counterpart to this generation skill).
- `.claude/workflows/napl-port-unarysim.js` - the batch sweep whose RTL phase delegates to this skill.
