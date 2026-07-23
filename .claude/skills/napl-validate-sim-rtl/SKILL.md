---
name: napl-validate-sim-rtl
description: >-
  Validate that a napl kernel's Python functional model and its Verilog RTL produce
  identical outputs on the same inputs taken from test_<kernel>.py. Use whenever the
  user wants to confirm a kernel's generated RTL matches the simulator/Python model,
  check sim-vs-RTL or hardware-vs-model equivalence, verify the Verilog is functionally
  correct, run the golden-vector co-simulation against the test's inputs, or asks "does
  the RTL match the kernel", "validate the Verilog for mul_and", "is the shiftreg
  hardware correct", "check sim against RTL for add_any", even without the word
  "validate". The bar is bit-exact: the RTL output must equal the Python model output
  cycle-for-cycle on the streams the test encodes. Sibling to napl-gen-rtl (which generates the
  RTL this skill checks), napl-validate-unarysim (model vs the UnarySim reference), and the
  napl-port-unarysim workflow's RTL phase. Validate-only: if a kernel has no RTL
  yet, report that and point to napl-gen-rtl / napl-port-unarysim rather than generating it here. Every run
  leaves a durable summary row in reports/napl-validate-sim-rtl-report.md (kernel, RTL module,
  PASS/FAIL, vectors, regimes, reset checks) so sim-vs-RTL fidelity is tracked over time.
---

# Validate a napl kernel's Python model against its Verilog RTL

## Why this exists

napl lowers the same kernel two ways: a Python functional simulation that runs over
timesteps, and a synthesizable Verilog RTL counterpart under
`src/napl/implementation/<op>/`. They are only faithful if they compute the **same bits**:
one Python `forward()` timestep equals one `posedge i_clk`, and the RTL output must equal
the model output cycle-for-cycle. The repo already has the co-simulation engine for this,
`make test OP=<op>` (generate golden vectors from the Python model, compile RTL + testbench
with Icarus Verilog, simulate, and print PASS only on a full match). What this skill adds is
the missing constraint the user cares about: the golden vectors must come from **the same
inputs `test_<kernel>.py` uses**, not from whatever ad hoc stream the generator happened to
hardcode. So the validation proves the RTL matches the model on the regimes the Python test
actually exercises (both polarities, the known-answer corners, the reset transient), not on
an unrelated sweep.

It distills one rule from each sibling:

- **From the napl-port-unarysim RTL phase:** golden/expected vectors ALWAYS come from the napl
  Python model, never a hand-written truth table; the testbench prints PASS only on a full
  bit-exact match; `i_rst_n` low maps to the model's `reset()`; one `forward()` equals one
  `posedge i_clk`.
- **From napl-validate-unarysim:** feed **identical inputs** to both sides by mirroring the
  test's exact encoder configs (napl encoders are deterministic, so the same config and value
  reproduce the same spike stream); run in a subagent; on a mismatch, locate the diverging
  cycle and say which side is right, do not hand-wave.

The deliverable is a verdict: kernel, RTL module(s) compared, the test-derived regimes used,
the bit-exact PASS/FAIL per polarity, and an explanation of any divergence.

## Execution model: always run the validation in a subagent

**Always delegate to one subagent; never run it in the main thread.** It is a multi-step job
(read the kernel + its test + its RTL, write a test-driven vector generator, run the
gen/compile/simulate co-sim, interpret) that emits a lot of transient output: vector dumps,
iverilog warnings, vvp logs, mismatch traces. Keeping that out of the main conversation is the
point. When invoked with a kernel target, the main agent only dispatches and relays:

1. Spawn **one** `general-purpose` subagent (`Agent`/`Task` tool) with a prompt like:

   > Validate the napl kernel `<target>`'s Python model against its Verilog RTL. Follow the
   > skill at `.claude/skills/napl-validate-sim-rtl/SKILL.md` exactly: read it, then execute
   > Steps 1-6 (resolve the kernel + its `test_<kernel>.py` + RTL dir; make `gen_<op>.py`
   > derive its golden vectors from the test's inputs/regimes; run `make test OP=<op>`;
   > require a bit-exact PASS; on FAIL locate the diverging cycle; then record the validation
   > summary row with `scripts/record_sim_rtl.py`). Return the summary block, the exact
   > `make test` result, and the recorded log row.

2. Relay the verdict. If the subagent reports FAIL, surface where it diverged (reset state,
   pp_delay alignment, polarity, or a genuine logic bug) rather than presenting a green pass.

**If you ARE that dispatched subagent**, ignore this section and execute Steps 1-5 directly.
Validate one kernel per subagent.

## Environment

- Run through the project env: `conda run -n napl python ...` and
  `conda run -n napl make test OP=<op>` (a bare `python` is the wrong interpreter). Heredocs
  piped through `conda run` swallow stdout, so write a `.py` file and run it, never inline a
  `python - <<EOF`.
- The co-sim needs the **Icarus Verilog** toolchain (`iverilog`/`vvp`) on PATH. If it is
  missing, report that the co-sim cannot run rather than guessing a result.
- Everything lives under `src/napl/implementation/<op>/{rtl,tb,gen,vec,build}/`; run `make`
  from `src/napl/implementation/`. `vec/*.vec` and `build/` are generated (gitignored).

## Workflow

### Step 1 - Resolve the kernel, its test, and its RTL

Pin down the napl class (e.g. `operation.mul_and`, `operation.shiftreg`), its source under
`src/napl/operation/`, its `tests/<subpackage>/test_<kernel>.py`, and its RTL directory
`src/napl/implementation/<op>/`. The op directory and `make test` OP name is the class name.

**Validate-only:** if `src/napl/implementation/<op>/` does not exist (no RTL yet), STOP and
report that the kernel has no RTL to validate, pointing the user to the **napl-port-unarysim**
workflow's RTL phase to generate it first. Do not generate RTL here; that is napl-port-unarysim's
job, and this skill checks an existing implementation.

### Step 2 - Read the test and the RTL contract

Read `test_<kernel>.py` for the inputs you must reuse: the codec configs (polarity, generator,
Sobol dim, timestep), the regimes it covers (both polarities, known-answer corners such as a
stream vs itself / vs its complement, edge inputs like all-zero), and how the op is wired
(encoder(s) -> op -> decoder). Read the RTL module's port list and the testbench's vector
format (`$fscanf` columns) so your generated vectors line up with what the testbench reads.
Note the op's `self.hw.pp_delay` (input-to-output latency in cycles) and what `reset()`
initializes, including non-zero reset state (e.g. `shiftreg` reloads `reg[i] = i % 2`, not
zeros); the RTL `i_rst_n` must reproduce exactly that.

### Step 3 - Make gen_<op>.py derive its vectors from the test's inputs

This is the core of the skill. Rewrite `<op>/gen/gen_<op>.py` so its golden vectors come from
the **test's** inputs instead of an ad hoc stream, then it stays reproducible via `make test`.
Key points:

- **Mirror the test's encoder configs** (the napl-validate-unarysim trick): napl encoders are
  deterministic, so building an encoder from the test's exact `codec_config` and feeding it a
  value reproduces the identical per-timestep spike stream the test sends the op. Drive the
  napl **op directly** with those streams (the op boundary, bypassing the decoder), recording
  per-timestep `(op inputs, op output)`. Start from `model.reset()` at `t=0` so the reset
  transient is part of the vectors.
- **Build the op with the test's sizing config, and make the RTL inherit it.** The op's size
  (`depth`, `width`, `scale`, `entry`, ...) is a Verilog `parameter`, not a baked-in constant
  (see the implementation spec). Build the Python model from the **test's** op config (e.g.
  `test_shiftreg.py`'s `shiftreg_config={'depth': 2}`), and from that SAME config emit
  `vec/<op>_params.vh` with one `` `define GEN_<PARAM> <value> `` per sizing field. The testbench
  `` `include "<op>/vec/<op>_params.vh" `` (iverilog resolves it relative to the compile cwd,
  `implementation/`) and instantiates the DUT with `<op> #(.PARAM(`GEN_PARAM)) dut (...)`, so the
  RTL is validated at the test's size, not a stale default. If the existing gen hardcodes a size
  (an old `localparam`/constant mirrored by a "must match" comment), this is exactly the drift to
  fix - replace it with the inherited parameter. (Real example: `shiftreg`'s RTL/gen carried
  `DEPTH=4` while the test used `depth=2`; the co-sim passed only because both wrong copies agreed.)
- **The RTL is a bit-serial scalar datapath; the test's inputs are vectorized.** The test
  feeds a large tensor (many parallel streams) only to gather SC error statistics. The RTL has
  one 1-bit datapath, so you cannot feed it a vector. Instead drive **representative scalar
  streams** that use the test's exact encoding: the test's known-answer corner values, a small
  sweep across the value range, and a few draws from the test's distribution, each encoded with
  the test's config over the test's `timestep` count. This is "the same input from the test"
  at the granularity the hardware accepts.
- **Cover both polarities** if the op has unipolar and bipolar variants (one RTL module per
  polarity postfix, per the implementation spec), emitting the per-polarity expected column the
  testbench reads (see `mul_and`'s vec format: `in_0 in_1 out_unipolar out_bipolar`).
- **Stateful ops need a reset between independent streams.** Each representative stream is an
  independent experiment, so the model must `reset()` between them and the testbench must pulse
  `i_rst_n` low between segments. If covering more than one stream requires the testbench to
  re-assert reset, update the testbench too (keep its PASS-only-on-full-match check intact).
- **Test that reset itself is identical, not just power-on output.** Output equivalence from a
  single power-on reset is necessary but NOT sufficient to prove reset is faithful: a register
  whose reset value is wrong but happens to be overwritten before it is read (a hidden
  circular-buffer index, a counter that only matters under a specific sequence) passes a clean
  stream while the reset state actually differs. For any stateful op, drive a stream long enough
  to dirty the internal state, then assert reset **mid-stream** (call `model.reset()` and have
  the testbench pulse `i_rst_n` low at that same cycle), then continue driving and require the
  post-reset outputs to keep matching. This proves reset returns BOTH sides to the same state
  from an arbitrary operating state, which is the real meaning of "reset is identical". Emit a
  reset marker in the vector stream and extend the testbench to re-pulse `i_rst_n` on it (a small
  extension to the single-reset testbench; keep the PASS-only-on-full-match check). For an op
  whose reset value is not directly observable in the next output, this mid-stream reset is the
  only way the co-sim can surface a wrong reset value at all.

Two live patterns to copy: `implementation/mul_and/gen/gen_mul_and.py` (combinational,
exhaustive input product) and `implementation/shiftreg/gen/gen_shiftreg.py` +
`shiftreg_tb.v` (stateful, a single per-cycle spike stream replayed after one reset). Adapt the
nearer one, swapping its ad hoc input source for the mirrored test encoders.

If `gen_<op>.py` already derives its inputs from the test wiring, confirm that and skip the
rewrite.

### Step 4 - Run the co-simulation

From `src/napl/implementation/`, run the standard engine:

```bash
conda run -n napl make test OP=<op>
```

It runs your test-driven `gen_<op>.py` to (re)write `vec/<op>.vec`, compiles `rtl/*.v` plus the
testbench with `iverilog -g2012 -Wall`, simulates with `vvp`, and the testbench prints `PASS`
only when **every** vector matches. The Makefile greps for `^PASS` and fails the build
otherwise. Bit-exact is the bar: spikes are 0/1, the RTL is a faithful gate-level implementation
of the model, so the outputs are identical or there is a real bug, never "close".

### Step 5 - Interpret and report

- **PASS** is the expected outcome: the RTL reproduces the model bit-for-bit on the test's
  streams. Report it with the regimes covered.
- **FAIL** is a finding, not a dead end. Read the testbench's `FAIL[n] ... got X exp Y` line
  to locate the **first** diverging cycle, then classify it:
  - **first few cycles only (cycle < pp_delay):** a reset-state mismatch. The RTL `i_rst_n`
    state does not match the model's `reset()` (e.g. reset to zeros where the model reloads a
    non-zero pattern), or the testbench did not assert reset from `t=0`.
  - **clean until a mid-stream reset, then diverges:** the RTL does not return to the model's
    `reset()` state from an operating state (a register not cleared on reset, or async vs sync
    reset semantics differing from `reset()`). This is the case a power-on-only check misses.
  - **a constant cycle offset:** a pp_delay misalignment. The model already encodes its latency,
    so vectors should align cycle-for-cycle; an offset means the RTL has a different register
    depth than `self.hw.pp_delay`, or the testbench samples on the wrong edge.
  - **one polarity passes, the other fails:** a polarity-branch bug (wrong gate: AND vs XNOR,
    or a missing inverse path).
  - **scattered mismatches:** a genuine logic divergence; say which side is right relative to
    the model (the model is the reference) and where in the RTL the gate logic differs.
Deliver the verdict as this exact **validation summary** block, so every run reports the same
fields (and Step 6 records them):

```
Validation summary
  Kernel:        operation.<name>
  RTL module(s): <op>(.v) [, <op>_unipolar / <op>_bipolar if split]
  Result:        PASS | FAIL          (bit-exact, make test)
  Vectors:       <matched>/<total>
  Polarities:    <unipolar, bipolar | bipolar | ...>
  Reset checks:  power-on (t=0 vs reset()) [+ mid-stream from dirtied state, for stateful ops]
  Regimes:       <corner values; distribution draws; depth; timestep; generator>
  Divergence:    <none | first-failing cycle + classification + which side is right>
```

Report honestly; a FAIL that exposes a real RTL bug is the skill working, not a failure of the
run. The model is the reference: when they differ, the RTL is wrong unless the model itself is.

### Step 6 - Record the validation summary (always)

Every validation must leave a durable, trackable row in **`reports/napl-validate-sim-rtl-report.md`**
so sim-vs-RTL fidelity is tracked over time (the same "record always" discipline
napl-validate-unarysim applies to its UnarySim log). Append the row with the bundled recorder
(it creates the file with a header on first use, dedupes, and keeps rows sorted by kernel):

```bash
conda run -n napl python .claude/skills/napl-validate-sim-rtl/scripts/record_sim_rtl.py \
    --kernel operation.shiftreg --rtl shiftreg \
    --result PASS --vectors "2048/2048" \
    --polarities bipolar \
    --reset "power-on + mid-stream" \
    --regimes "depth=4 T=256 sobol; corners {-1,0,+1} + 4 distribution draws" \
    --notes "non-zero i%2 reset; mid-stream reset from dirtied FIFO matched"
```

Record the outcome of EVERY run, PASS or FAIL (on FAIL, put the diverging cycle + cause in
`--notes` and set `--result FAIL`), so a known-failing RTL is visible in the log rather than
silently absent. Pass `--date YYYY-MM-DD` only to backfill; it defaults to today. Confirm the
recorded row to the user as part of the verdict.

## Bundled resources

- `scripts/record_sim_rtl.py` - append a uniform summary row to
  `reports/napl-validate-sim-rtl-report.md` (Step 6). Creates the file with a header on first use,
  escapes table-breaking pipes, dedupes identical rows, and keeps the log sorted by kernel then
  date so re-validations of one kernel group together.

## Gotchas (learned the hard way)

- **Identical inputs or it proves nothing.** The whole point is that the RTL sees the SAME
  per-cycle streams the test encodes. Mirror the test's encoder configs; do not invent a new
  random stream (the old gen's mistake this skill fixes).
- **Vectors come from the model, never a truth table.** The expected column is whatever the
  napl `forward()` emits on that stream. A hand-derived expectation would validate the RTL
  against your arithmetic, not against napl.
- **Reset from t=0, to the model's actual state.** Drive the model from `reset()` and have the
  testbench pulse `i_rst_n` low before the stream. The reset state is not always zero
  (`shiftreg` reloads `i % 2`); a wrong reset value fails on the opening cycles.
- **Power-on reset alone does not prove reset is identical.** A wrong reset value on a register
  that gets overwritten before it is read passes a clean stream. Assert reset MID-stream from a
  dirtied state (Step 3) and require continued match, so reset equivalence is tested from an
  arbitrary state, not only at `t=0`.
- **Bit-serial scalar, not the test's tensor.** The RTL has one 1-bit datapath. Drive
  representative scalar streams encoded with the test's config, not the test's wide tensor.
- **Validate at the test's size, via inheritance.** Build the model with the test's sizing config
  and have the RTL inherit it: gen emits `vec/<op>_params.vh` (`` `define GEN_<PARAM> ... ``) from
  that config; the tb `` `include ``s it (path relative to the compile cwd: `"<op>/vec/<op>_params.vh"`)
  and overrides the DUT parameter. A hardcoded RTL size that disagrees with `test_<op>.py` is a real
  finding, not a pass - it means you validated a configuration the test never runs.
- **pp_delay alignment.** One `forward()` equals one `posedge i_clk`; the model's per-tick
  output already includes its latency, so model vectors align with RTL cycles. A constant offset
  in the mismatches points at a pp_delay or sampling-edge bug, not a logic bug.
- **One module per polarity.** Per the implementation spec, distinct unipolar/bipolar variants
  are separate RTL modules with a single `_unipolar`/`_bipolar` postfix; emit and check the
  per-polarity expected column accordingly.
- **iverilog must be on PATH.** No toolchain means no co-sim; say so rather than reporting a
  result you did not run.

## References

- `RULE_RTL.md` (repo root) - the RTL rules: layout, commands, naming, combinational vs clocked,
  the `i_clk`/`i_rst_n` and reset-state contract.
- `src/napl/implementation/mul_and/` - canonical combinational example (gen, tb, rtl).
- `src/napl/implementation/shiftreg/` - canonical stateful example: per-cycle stream, non-zero
  reset, the `forward()`-equals-`posedge` and `reset()`-equals-`i_rst_n` mapping.
- `tests/<subpackage>/test_<kernel>.py` - the source of the inputs and regimes to reuse.
- `.claude/skills/napl-validate-unarysim/SKILL.md` - sibling skill; reuse its mirror-the-encoder
  pattern for identical inputs.
- `.claude/skills/napl-gen-rtl/SKILL.md` - the producer sibling that generates the RTL this skill
  validates (generate then validate).
- `.claude/workflows/napl-port-unarysim.js` - the batch sweep whose RTL phase generates (via
  napl-gen-rtl) then re-verifies (via this skill) each op's RTL.
- `scripts/record_sim_rtl.py` and `reports/napl-validate-sim-rtl-report.md` - the recorder and the
  durable log this skill appends a summary row to on every run (Step 6).
