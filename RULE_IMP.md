# NAPL hardware rules

This file is the canonical design and verification policy for `src/napl/imp/`, plus its directory layout and commands. Where `src/napl/sim/operation/` is the *functional* Python model, the `src/napl/imp/` tree is its *hardware* counterpart: Verilog-2001 implementations of the NAPL stochastic-computing operations, one synthesizable module per concrete operation variant, each verified against the Python model with golden-vector co-simulation. `src/napl/imp/module/` extends the same flow to `sim/module` classes, whose lane-replicated RTL composes those operation circuits. Policy comes first; the layout and flow mechanics follow.

## Design contract

General Verilog style (Verilog-2001, `i_`/`o_` port prefixes, `i_clk`/active-low `i_rst_n`, one module per file with the file name equal to the module name, lint-clean) follows the unarylab-research plugin's verilog-rules hook, which injects the full rules when a Verilog file is edited. Separate every module, function, or task declaration from adjacent declarations or code in the same scope with exactly two blank lines.

Three numbered lists in this file run their own sequences: the design contract below, the per-change list under "Verification gates", and the module layer. Cite them as *design contract rule N*, *verification step N*, and *module-layer rule N* respectively. This paragraph carries no number of its own and is cited as the *citation-scope rule*.

1. **Variants and names.** Use `_unipolar` and `_bipolar` modules whenever polarity changes the circuit; do not select polarity-dependent logic with a parameter. An operation whose circuit is identical across supported polarities uses the bare operation name. The operation folder name and RTL module base name equal the `sim/operation` module name verbatim; the RTL module is `<op>` or `<op>_<polarity>`. A non-polarity Python configuration field that selects between circuits, such as `scaled`, is an integer Verilog parameter fixed at elaboration, never a runtime input. A `forward()` argument that does arrive at runtime, as `scale` does on `div_scale_dyn`'s and `add_scale_dyn`'s `i_scale`, is an unsigned integer bus of `ceil(log2(scale_max + 1))` bits, and `mapping.yaml` gives both operations an RTL form only at `fracwidth == 0`, so the port encodes whole units and the caller must pass an integer `scale`. The Python model takes any positive float up to `scale_max` and quantizes it to `2 ** -fracwidth`, so a non-integer request runs there and has no hardware meaning; each generator asserts the scales in its own schedule are integers, and nothing checks a caller outside that path. Distinct operations or port sets are separate operations with their own folders (e.g. `mul_gaines` vs `mul_ugemm`).
2. **Sizing parameters.** Sizing fields such as depth or width remain Verilog parameters derived from the Python configuration. Declare each with a default documenting the configuration it was generated for, and derive every bus width and internal constant from the parameters so the module is correct at any size.
3. **Spikes and cycles.** Spikes are 1-bit wires, one spike per stream per clock. Each RTL port name is its `i_` or `o_` prefix plus the corresponding Python `forward()` argument name verbatim, with no aliases. Each module is one scalar circuit per operation variant; vectorization across lanes is replication handled by higher-level blocks, not baked into the operation. One Python `forward()` timestep corresponds to one `posedge i_clk`.
4. **Reset.** Every register must return to the exact post-`reset()` state of the Python model when `i_rst_n` is asserted low. The correct state is not necessarily zero.
5. **Timing.** Record verified input-to-output latency in the Python operation as `self.hw.pp_delay = ...`: zero for a combinational path and one cycle per register stage. If latency depends on a sizing field, derive both RTL latency and `pp_delay` from that field (e.g. `shiftreg`'s `pp_delay == config['depth'] == DEPTH`).

   Per-corner STA data belongs in `self.hw.timing`, keyed by the MCMM scenario `node/process/voltage/temp/rc/mode`. A combinational operation records its full through-delay in `cp_delay` with zero `ir_delay` and `or_delay`; a registered operation records all three timing components.

6. **RTL coverage.** Every operation with `streaming = True` requires a verified RTL counterpart; classes with `streaming = False` (non-streaming binary-domain: `relu_fxp`, `sigmoid_fxp`, `tanh_fxp`, `round_fxp`, the FXP modules, and `butterfly_fp`; plus the non-streaming HUB wrappers) have none. Every streaming operation has its `operation/<op>/` folder, so the RTL backlog is empty. Three module RTLs carry an outstanding revision rather than a missing folder: each takes its operands as encoded spikes on ports and must instead hold the operand encoders in its own RTL.

   - `linear_mix`: `i_weight` and `i_bias` are spikes (`module/linear_mix/rtl/linear_mix_unipolar.v:29-30`, `linear_mix_bipolar.v:30-31`).
   - `conv_mix`: `i_weight`, `i_bias`, and the bipolar pad stream `i_pad_bits` are spikes (`module/conv_mix/rtl/conv_mix_unipolar.v:46-47`, `conv_mix_bipolar.v:46-48`), so the pad encoder is external too.
   - `mgu_hard_mix`: `i_weight_f`, `i_bias_f`, `i_weight_n`, and `i_bias_n` are spikes (`module/mgu_hard_mix/rtl/mgu_hard_mix_bipolar.v:72-75`).

   `linear_ugemm`, `conv_ugemm`, `linear_gaines`, and `conv_gaines` take held numeric codes on the same ports and encode in hardware, so they need no such revision.
7. **Input handling.** Every streaming operation's RTL consumes its `i_*` data inputs combinationally in the arrival cycle: registers hold internal state only and never re-register a raw input before its first use. For delay and storage elements whose defined function is capturing the input (`dff`, `shiftreg`, `jkff`, `square_dff`'s internal delay), that capture is the first use.
8. **Explicit imports.** Python generator and testbench scripts under `src/napl/imp/` use explicit imports and do not use `from x import *`. Package `__init__.py` re-exports may use star imports.
9. **Seeded sequences.** Every configuration a generator builds a model or encoder from must set `seed` whenever `generator` is `sys`. An unseeded `sys` sequence is drawn fresh per instance, so the emitted ROM would not match the sequence a separately constructed model draws, and the co-simulation cannot reveal the mismatch because it builds a single model. Nothing enforces this: `encode.__init__` and `gen_num_seq()` both accept an unseeded `sys` configuration, so each generator must set the seed itself.

## Per-operation structure

Each concrete operation lives in `operation/<op>/` with `rtl/`, `tb/`, `gen/`, `vec/`, and `build/` subdirectories. The testbench is `<op>_tb.v`; the generator is `gen_<op>.py`.

The generator builds the NAPL Python operation from one configuration, emits expected outputs from that model, and writes sizing values to `vec/<op>_params.vh`. The testbench includes that header and applies the same values to the RTL parameters. Do not hardcode a second copy of sizing values in the testbench or RTL. Choosing that configuration smaller than the class's own test, so a clamp lands inside a vector row, is licensed by module-layer rule 9 on the terms stated there.

`vec/*.vec`, `vec/*_params.vh`, and `build/` are generated and must remain untracked.

## Verification gates

Golden-vector co-simulation is the source of functional truth. Expected outputs must come from the NAPL Python model, never from a hand-written truth table. A single `<op>.vec` file carries the expected outputs for every variant; the testbench instantiates all of an operation's variant modules, must compare every output on the corresponding cycle, and the verification command must exit nonzero on any mismatch.

Every Python `test_*` function used by this verification flow starts with a concise one-line docstring that states the behavior it verifies.

RTL co-simulation uses the fidelity-scale input shape from the streaming suite's `make_values`. The `make_random_perf_values` workload times the Python model only: it must never be used to build a vector file or to drive a co-simulation.

For each RTL-backed change, cited as *verification step N*:

1. Run the operation's Python test under the rules in [RULE_SIM.md](RULE_SIM.md).
2. From `src/napl/imp/`, run `conda run -n napl make test OP=<op>` for an operation or `conda run -n napl make test MODULE=<module>` for a module, and require an exit status of zero and a full-match `PASS`. The two selectors are exclusive; giving both is an error.
3. For a stateful operation, verify the first post-reset cycle and a reset asserted after state has changed, then replay the same inputs.
4. Verify the measured cycle latency against `self.hw.pp_delay`, including every supported sizing configuration used by the test.
5. Report the command, configuration and sizing parameters, vector count, reset cases, observed latency, expected `pp_delay`, and exit status.

A claim that a branch is never taken, which is what a dead clamp arm or an
unreachable rail amounts to, needs a **branch oracle**: keep the condition exactly as
written, change what the taken arm produces, and rerun the co-simulation. A green run
then means the branch was never entered, and a red one names the rows that enter it.
Halving the constant is not that test. It moves the condition as well as the result,
so a green run may mean only that the halved condition is still unreachable and a red
run may mean only that the halved condition became reachable while the original stays
dead. Both directions of that error have occurred in this tree, so any dead-or-live
classification resting on a halved constant is unsound until an oracle repeats it.
A branch the oracle shows is reachable but that no vector row enters is live but
unexercised: it is stimulus that is missing, not logic, and the fix is a stimulus that
enters it, not a removal.

For each changed unit, run a focused co-simulation: `conda run -n napl make test OP=<op>` for an operation or `conda run -n napl make test MODULE=<module>` for a module, from `src/napl/imp/`, and require an exit status of zero. That focused co-sim of every changed unit is the per-change verification. A full `conda run -n napl make sweep` runs only when the user explicitly calls for it. When invoked, `make sweep` runs the mapping floor, the guards suite, the reference-form equivalence walk, the translate pytest, every operation target, and every module target, discovering the unit target list from the `operation/*/rtl` and `module/*/rtl` directories, so a folder added to either tree is covered without editing the `Makefile`. It runs the whole list rather than stopping at the first failure, prints a `*** SWEEP FAILED:` summary naming every red target, and exits nonzero when that list is non-empty. Each unit's own log stays in `<layer>/<unit>/build/sim.log`, and the floor, guards, equiv, and translate logs stay in `src/napl/imp/build/` and are printed when their step fails.

The floor, `make floor`, closes the gap the glob leaves: a deleted or renamed unit folder drops out of the glob, so the sweep would otherwise report every remaining target green. `sweep_floor.py` derives the expected unit set from `mapping.yaml`, the authoritative registry, and fails naming any registered unit the glob missed, any discovered folder with no mapping entry, and an empty target list. A registered folder that still exists but no longer holds usable RTL is the same hole one level down, so the floor also requires each entry's `<rtl_module>.v` to exist under some `<layer>/*/rtl/` directory, and each discovered folder's `rtl/` to hold at least one `.v` file. Both checks are needed: the per-entry check searches every `<layer>/*/rtl/` directory rather than the entry's own folder, so a registered module name found in a sibling folder satisfies it while the entry's own folder sits empty, and only the per-folder check sees that folder emptied.

There is no local sweep gate on commits. The focused co-simulations of the changed units are the per-change check, and a full `make sweep` runs only when the user explicitly calls for it or in a CI job.

The equivalence step, `make equiv`, runs `test_equiv.py`, which covers the units whose outer body inlines another operation's logic while that operation also exists as a standalone module. It drives each host copy beside its standalone reference over 200k random cycles with periodic resets, compares them every cycle, elaborates a copy carrying a sizing parameter at depths above the one its generator commits, and, for the copies that drop a clamp arm, drives every input sequence of length 16 from reset and requires the tapped pre-clamp sum to stay inside the bounds the dropped arm would have enforced. It is a random walk, not a proof: a walk visits only the states its stimulus happens to reach, so a divergence reachable only on a rare state sequence passes it, and the probe is exhaustive only over sequences of that length.

The translate step, `make translate`, runs `tests/syn/test_translate.py`, the only gate on `mapping.yaml` and `syn/translate.py`, which carry no RTL of their own and so belong to no unit target.

`make install-hooks` points `core.hooksPath` at the repo's versioned `.githooks/` directory. The `pre-commit` hook there is a no-op and does not gate commits: the per-change check is the focused co-simulation of every changed unit, run explicitly.

## Module layer

`module/<name>/` holds the hardware counterpart of a `sim/module` class and mirrors the operation layout exactly: `rtl/`, `tb/`, `gen/`, `vec/`, and `build/`, with testbench `<name>_tb.v` and generator `gen_<name>.py`. The design contract and verification gates above apply unchanged; the following are the module-layer specifics, cited as *module-layer rule N*.

1. **Composition.** Module RTL composes operation-layer scalar circuits, and a module whose Python class composes other `sim/module` classes instantiates their RTL in turn. A module instantiates the `operation/*/rtl/` or `module/*/rtl/` module whose function it needs instead of restating that logic, so each layer stays the single implementation of every circuit it owns.
2. **Vectorization.** A module covers many lanes: one lane per tensor position the Python class produces. Lanes are `generate`-`for` replications of the scalar circuit, and the lane count is a Verilog parameter (`LANES`) derived from the Python input shape and geometry. Lane `l` occupies the low-to-high slice `[l*<lane width> +: <lane width>]` of each vector port.
3. **Ports.** Each port name is its `i_` or `o_` prefix plus the corresponding Python `forward()` argument name verbatim, as in design contract rule 3. Spikes remain 1-bit per lane per clock, so a port carrying `LANES` lanes is a packed vector of them.

4. **Golden vectors.** Expected outputs come from the Python module model in `sim/module/`, driven by the encoder configuration and the fidelity-scale input shape of the class's `test_<name>.py`. The generator flattens the model's per-timestep input and output tensors into the lane order the RTL ports use.
5. **Timing.** One Python `forward()` timestep corresponds to one `posedge i_clk`, the same as the operation layer, and the testbench checks the observed latency against the class's `self.hw.pp_delay`.

6. **Elaboration guards.** A sizing restriction the RTL cannot check at runtime is enforced at elaboration by instantiating an undefined module inside a `generate` `if`, so the build fails with that module's name. The guard module is named `ERROR_<base>_<restriction>`, where `<base>` is the polarity-stripped operation or module base name, so every polarity variant of one restriction reports the same name. Guards are elaboration-time constant folding and carry zero area, so they are exempt from the "no excess sanity-checking logic" rule, which targets runtime logic. Write the guard condition so it does not overflow 32-bit integer arithmetic: compare widths (`WIDTH - 1 < clog2(ENTRY + 1)`) rather than the values themselves (`2 ** (WIDTH - 1) <= ENTRY`).

    The `GUARDS` table in `imp/test_guards.py` is the authoritative record of which RTL module carries which restriction; read it there rather than from a list here. Each row elaborates a violating parameter set and requires `iverilog` to fail with the guard's name, plus a legal set right below the boundary that must still elaborate. `make guards` runs the table, and so does every `make test` target, so a guard that is deleted or written backwards fails that test and fails every unit test with it. Each restriction is also stated in the RTL header banner of the module that carries it, and every restriction `mapping.yaml` can violate is mirrored there as a `requires` condition, so translation rejects the configuration before elaboration does.

    A restriction a module inherits from a circuit it composes is carried as a `requires` condition without a second guard, the composed module's own guard being the elaboration-time check.

    A guard term whose violating parameter set cannot be elaborated has no row: `add_scale_*`'s `ENTRY` term needs a bus of hundreds of millions of lanes, so `mapping.yaml`'s `requires`, evaluated in Python, is the only check on it.

7. **Golden-column widths.** `%b` zero-extends a short vector silently, so a testbench that scans a golden column straight into a sized `reg` passes at the wrong lane count. Scan each vector column as `%s`, check its character count against the port width, and convert the characters to bits (`token_len`, `token_bits`, and `check_width` in `conv_mix_tb.v` are the pattern to copy). A testbench copied from an existing module testbench inherits these three helpers with the rest of the pattern.

    A count-bus column is that one pattern at a different width: the column is the lane counts packed exactly as the port packs them, `LANES * COUNT_W` characters highest bit index first, and `check_width` compares against `LANES * COUNT_W` instead of `LANES`. That assertion is an echo of `COUNT_W`, not a check on it: the generator's own `count_width()` sizes both the column and the elaborated parameter, so the comparison fires on a wrong lane count rather than a wrong count width. `COUNT_W` is pinned by the elaboration guard against `clog2(ENTRY + 1)` and by `check_mapping()` against the `mapping.yaml` expression.

    The token registers are one character wider than the widest golden column (`MAX_CHARS = <widest column> + 1`). `$fscanf` with `%s` truncates to the token width, so a token sized exactly to the widest column saturates at the expected length and an over-long column of that width passes the check. The spare character makes an over-long column read back long and fail.

8. **Vector count.** The generator emits its row count as `` `define GEN_VECTORS ``, and the testbench requires the number of rows it consumed to equal it, so a vec file that lost or gained rows fails instead of passing on the rows it still holds. The row loop ends on the first `$fscanf` that does not return every column, rather than looping on `$feof`, which spins when a scan stops consuming. The check is unconditional: a generator that stopped emitting the define fails the build rather than skipping the check.

9. **Parameter observability.** Every elaboration parameter a unit carries must be observable in at least one vector row: corrupting the `` `define GEN_<PARAM> `` in the generated header, without regenerating the vectors, must make `make test` fail. Geometry and lane counts are observable through the golden-column width assertions and the elaboration guards. A headroom parameter is not, so a unit that carries one needs a stimulus block that drives the state it bounds onto its clamp and then reads the stored state back out, since only a clamped excursion changes a compared output.

    That mechanism is what licenses a sizing deviation, and nothing else does. A generator may elaborate a unit at a narrower width, shorter sequence, or smaller fan-in than the class's own test uses, when at the test's size no vector row can reach the clamp: the narrow size is what makes the headroom parameter observable at all. The deviation holds only while the narrow size is a configuration the Python model accepts, the generator builds its golden model from that same configuration so the compared outputs still come from the model, and the generator asserts how far the excursion went against the clamp the elaborated size sets. A size chosen for any other reason, runtime included, is not covered by this rule.

    `linear_mix`, `conv_mix`, `linear_ugemm`, `conv_ugemm`, and `mgu_hard_mix` carry such a block. It rails the weight and bias operands, charges the accumulator over one input block and drains it over the next, and asserts in the generator how far the excursion went against the clamp the elaborated width sets. An explicit `scale` below the fan-in is what lets the accumulator move at all, so each of them elaborates one arm with such a scale: `conv_mix`'s geometry `d`, `conv_ugemm`'s geometry `d`, `linear_mix`'s fifth and sixth configurations, `linear_ugemm`'s fifth and sixth.

    The clamp acts on the pre-carry sum. `add_scale.forward()` adds the centered partial sum, clamps that sum to `[acc_min, acc_max]`, compares the clamped value against `scale_raw`, and only then subtracts a carry, so the quantity a width bounds is the sum before the carry and the quantity that is stored and compared on the next timestep is the sum after it. The accumulator holds raw units of `2 ** -fracwidth`, so its bipolar offset is `(entry * 2 ** fracwidth - scale_raw) / 2` and the width bounding it is `intwidth + fracwidth`. `mapping.yaml` requires `fracwidth == 0` of every RTL-backed configuration, where a raw unit is a unit, `scale_raw` is `scale`, and `WIDTH` is `intwidth`; the paragraphs here and below are written at that setting and their arithmetic is unchanged by the fractional form. With `scale == entry` the bipolar offset `(entry - scale) / 2` is 0, every addend is non-negative, and the *stored* value is confined to `[0, entry - 1]`: a carry subtracts `scale == entry`, so a state at or below `entry - 1` returns to at most `entry - 1` however large the partial sum. The *clamped* quantity in that same regime reaches `2 * entry - 1`, the largest stored state plus a full partial sum, so no clamp is reachable inside that range only while `2 ** (WIDTH - 1) - 1 >= 2 * entry - 1`. The elaboration guard is the weaker `2 ** (WIDTH - 1) > entry`, so a width one bit above the guard's floor can clamp a `scale == entry` arm.

    Both rails need charging, and they are not symmetric. In unipolar the offset is 0 and a carry only subtracts down to 0, so the accumulator never goes negative and `ACC_LO` is unreachable there by construction rather than for want of stimulus. Only a bipolar accumulator with `scale` below `entry` moves down, by the offset per cycle. With a partial sum `p`, an accumulator that is emitting climbs at `p - offset - scale` per cycle and drains at `offset - p` (bipolar) or `scale - p` (unipolar), so a block charges with `p` railed to `entry` and drains with `p` railed to 0. `linear_mix`'s, `conv_mix`'s, and `mgu_hard_mix`'s blocks reach both clamps: they rail the bias low for the negative block, since a railed-high bias keeps one addend on every cycle. `mgu_hard_mix`'s gate accumulator runs at `scale` 1 over `entry` 8, so it falls at the offset 3.5 per silent cycle and reaches -512 in 147 of the cycles its `mul_ugemm` budget allows. Its output adder is a second accumulator at `entry` 3 and `scale` 1, which drifts by at most 1 per cycle: no run inside that budget moves it past 256, so both of its clamps are unreachable and are recorded as such.

    A `mul_ugemm` unit multiplies that excursion by a cycle budget. Its sequence indices have no modulo and the ROM holds `2 ** SEQ_WIDTH` entries, and every timestep reads the sequence whether or not it advances an index, so a reset-to-reset run holds at most `2 ** SEQ_WIDTH` spiking cycles and, in bipolar, `2 ** SEQ_WIDTH` silent ones. A run that spends its whole spiking budget has no cycle left to drain in, so a block that charges and then drains holds at most `2 ** SEQ_WIDTH - 1` charging cycles.

    With a partial sum railed at `p` the pre-carry sum after `n` charging cycles is `(p - offset - SCALE) * n + SCALE`, where `offset` is 0 in unipolar and `(ENTRY - SCALE) / 2` in bipolar, the `fracwidth == 0` form of the sim offset above. Railing every operand high gives `p = ENTRY`; a block that rails the bias low instead, as `conv_ugemm`'s geometry `d` does to reach the negative side, spikes every addend but that one and gives `p = ENTRY - 1`. At `p = ENTRY`, so over the full budget the reachable peak is `(ENTRY - SCALE) * 2 ** SEQ_WIDTH + SCALE` in unipolar and `(ENTRY - SCALE) / 2 * 2 ** SEQ_WIDTH + SCALE` in bipolar, maximized at `SCALE = 1`. Where `SCALE > ENTRY` the per-cycle term is negative and the expression does not apply: the accumulator then hovers below `SCALE + p - offset` rather than growing with `n`, so the peak is that hover bound and never a negative number. Each form above is the pre-carry sum, which is what a width clamps; the stored value is that sum less one carry. Both were checked against an instrumented `add_scale` over 22 parameter points spanning `SCALE < ENTRY`, `SCALE == ENTRY`, and `SCALE > ENTRY`, in both polarities.

    `conv_ugemm` is over the boundary at its elaborated `SEQ_WIDTH` of 8: even at `ENTRY` 5 the unipolar peak is `4 * 256 + 1 = 1025`, past the 1023 clamp of an 11-bit accumulator. Reaching it costs the entire spiking budget, and a stimulus that shows a clamp must also drain, which leaves 255 charging cycles and a peak of 1021, so `ENTRY` 5 is pinnable in arithmetic and not in a vector row. Geometry `d` is elaborated at `ENTRY` 13 and `SCALE` 6 instead, on weights railed to the top code and a bias railed to 0: its lanes charge at 6 per cycle to a pre-carry 1086 over 180 cycles and drain at 6 per cycle, so an 11-bit clamp runs out of charge 10 cycles before the elaborated 12-bit one and the compared column differs. Its bipolar arm moves at the offset 3.5 per cycle, which the same budget stops at 450 and -497, so neither bipolar clamp is reachable at this `ENTRY` and `SEQ_WIDTH` and both are recorded as unreachable rather than covered. The generator asserts each of those four reaches and requires the block to part from a shadow model one bit narrower.

    `linear_ugemm` is over that boundary at `ENTRY` 17 and carries the block: its unipolar scale-5 arm climbs at 11 per cycle and reaches the 2047 clamp in 186 of the 255 cycles available. Its bipolar arm climbs and drains at the offset, 6 per cycle, which the same budget stops at 1275 and -1530: short of the elaborated clamps, past the 1023 and -1024 of a width one bit narrower, which is the corruption the block has to catch. The generator asserts the unipolar arm reached its clamp exactly, requires each bipolar excursion to pass the next narrower clamp, and requires the block to part from a shadow model one bit narrower.

The implemented module folders are `avgpool2d_ugemm`, `linear_mix`, `linear_gaines`, `linear_ugemm`, `conv_mix`, `conv_gaines`, `conv_ugemm`, and `mgu_hard_mix`, each named for the simulation class it implements. `avgpool2d_ugemm` has one variant, `mgu_hard_mix` is bipolar only, and the rest have one RTL variant per polarity. The HUB wrappers `linear_ugemm_hub`, `conv_ugemm_hub`, and `mgu_hard_mix_hub` have no RTL counterpart: each is a non-streaming wrapper (`streaming = False`) that runs the streaming inner cell for the whole internal timestep loop.

The remaining `sim/module` classes are outside the module layer's scope and have no RTL counterpart:

| Class | Scope |
| --- | --- |
| `linear_fxp` | Plain binary fixed-point datapath, no unary circuit. |
| `conv_fxp` | Plain binary fixed-point datapath, no unary circuit. |
| `mgu_hard_fxp` | Plain binary fixed-point datapath, no unary circuit. |
| `round_fxp` | Plain binary fixed-point datapath, no unary circuit. |

`linear_gaines_unipolar` and `linear_gaines_bipolar` implement the two polarity variants of `linear_gaines`. `SCALED` selects the unipolar Gaines adder at elaboration; the bipolar circuit supports only scaled addition, matching the simulation constructor's rejection of non-scaled bipolar data. `conv_mix` and `conv_ugemm` wire the im2col patch in hardware, so their input port carries the NCHW spike tensor. In `conv_mix_bipolar` a tap outside the input reads `i_pad_bits`, the pad spike the Python model encodes on its own Sobol dimension. The unipolar variant carries no such port: a unipolar zero pad is a constant zero spike, and the model builds a pad stream only for a bipolar stream with nonzero padding, so `mapping.yaml` maps `pad_bits` to no port for it and `tests/syn/test_translate.py` checks each entry's ports against the RTL header it names.

`conv_ugemm` generates its weight, bias, and pad streams inside the RTL, matching the Python class's `internal_encode = True`: the weight and bias ports carry held fixed-point codes rather than spikes, and there is no pad port. Its bipolar pad is the model's alternating 0/1 stream, built from a `jkff` held at J = K = 1, whose reset state is the model's opening `0`; its unipolar pad is a zero spike. A padded tap advances the sequence index of the `mul_ugemm_*` cell it drives, so it is never a don't-care. The Python model shares one sequence-index pair per (spatial output position, patch tap) across the output channels, because the patch spike broadcasts over them; each RTL lane holds its own pair, and every copy is advanced by the same patch spike, so the copies stay equal and the outputs are bit-exact with the shared-index model.

### `mapping.yaml`

One entry per RTL variant, keyed as:

| Key | Meaning |
| --- | --- |
| `rtl_module` | Verilog module name, and the RTL file base name. |
| `layer` | `operation` or `module`; the `imp/` subtree holding the RTL. Optional, defaulting to `operation`. Any other value is an error. |
| `sim_module` | Path of the Python class, `sim/operation/<op>.py` or `sim/module/<name>.py`. |
| `inputs` / `outputs` | Python `forward()` name to RTL port name, `null` for an argument with no port. |
| `parameters` | Verilog parameter name to a restricted expression, resolved in order so a later expression may read an earlier parameter. |
| `requires` | Optional conditions that must hold, evaluated after `parameters`; a false one raises `TranslationError`. |

Expressions run under the restricted evaluator in `syn/translate.py`, not `eval`: arithmetic, comparisons, `and`/`or`/`not`, conditional expressions, indexing, and the functions `ceil`, `floor`, `log2`, `len`, `int`, `get`, `shape`, and `area`. Names available are `config`, the node's `input` shape, `dim`, `SEGMENT`, every top-level key of `config`, and the parameters already resolved for this entry.

The node `config` for a module node carries the class's constructor arguments under their `__init__` names, so a class taking a `config=` mapping nests it as `config['config']`. The caller also supplies `lanes`, the number of pooled or tiled output positions the instance elaborates; translation injects nothing.

Each `gen_<name>.py` translates its own mapping entry and asserts the resolved parameters equal the ones it used to build its vectors, for every configuration the testbench elaborates. This is mandatory: it makes `make test MODULE=<name>` the gate on the mapping entry as well as on the RTL, so a wrong expression fails the co-simulation instead of passing unnoticed.

That assertion is one-directional. It catches an expression that resolves to something other than the value the vectors were built from, which is the failure mode for a derived expression such as `SEQ_WIDTH` or `SCALE`. For a pass-through parameter it proves nothing beyond that the expression reads the key it was meant to read, because the generator and the mapping read the same source: `LANES: config['lanes']` checked against the generator's own `LANES` is an echo. Such a parameter is only independently checked when the hardware itself is driven by it, as `LANES` is by the golden-column width assertion in the module testbench.

An accumulator width is the parameter this leaves open, which module-layer rule 9 covers: `linear_mix`, `conv_mix`, `linear_ugemm`, `conv_ugemm`, and `mgu_hard_mix` close it with a saturation block, and `add_ugemm_*`'s `ACC_WIDTH` is a derived operation-layer width whose vectors never drive the accumulator near its bound. A width that is larger than the model's stays invisible everywhere: the extra headroom changes no compared output.

The `napl-gen-rtl` skill generates and verifies operation folders only. Module folders are written by hand under this file.

## Directory layout

The tree contains 57 implemented operation folders under `operation/` and eight module folders under `module/`, plus the generic `Makefile`, the shared golden-vector helper `operation/_gen_common.py` (`encode_value`, `pair_streams`, ...), `test_guards.py`, the elaboration-guard test, `sweep_floor.py`, the mapping-registry floor `make sweep` runs first, and `test_equiv.py`, the host-vs-standalone equivalence walk. Implementations are added one operation folder at a time, either by hand following this file or via the `napl-port-unarysim` workflow (which generates each operation's RTL + testbench + generator and verifies it against the Python model). `mul_gaines` (unipolar = AND, bipolar = XNOR) is the canonical example referenced throughout.

Each unit owns a folder holding its RTL, testbench, generator, and generated artifacts:

```
imp/
├── Makefile                  # make test OP=<op> | make test MODULE=<module> | make sweep
├── test_guards.py            # make guards -- every elaboration guard, violating and legal
├── sweep_floor.py            # make floor  -- swept targets must match the mapping.yaml registry
├── test_equiv.py             # make equiv  -- inlined copy vs standalone reference, by random walk
├── mapping.yaml              # sim class -> RTL module, ports, and parameter expressions
├── operation/
│   ├── _gen_common.py        # shared golden-vector helpers (encode_value, pair_streams, ...)
│   └── <op>/                 # one folder per operation, e.g. mul_gaines/
│       ├── rtl/              # hand-written RTL modules, one per variant
│       │   ├── <op>_unipolar.v
│       │   └── <op>_bipolar.v
│       ├── tb/               # <op>_tb.v -- self-checking testbench
│       ├── gen/              # gen_<op>.py -- golden vectors + param header FROM the napl Python model
│       ├── vec/              # generated <op>.vec and <op>_params.vh (gitignored)
│       └── build/            # compiled sim + log (auto-created, gitignored)
└── module/
    └── <module>/             # one folder per module, e.g. avgpool2d_ugemm/
        ├── rtl/              # lane-replicated RTL composed of operation-layer circuits
        ├── tb/               # <module>_tb.v -- self-checking testbench
        ├── gen/              # gen_<module>.py -- golden vectors + param header FROM the napl Python model
        ├── vec/              # generated <module>.vec and <module>_params.vh (gitignored)
        └── build/            # compiled sim + log (auto-created, gitignored)
```

## Running a test

Requires the Icarus Verilog toolchain (`iverilog` / `vvp`) and the `napl` conda env (the vector generator imports from the `napl` package):

```bash
conda run -n napl make test OP=<op>          # operation layer
conda run -n napl make test MODULE=<module>  # module layer
conda run -n napl make guards                # every elaboration guard, all units (also run by make test)
conda run -n napl make sweep                 # explicit full sweep: floor, guards, equiv, translate, every operation, every module
conda run -n napl make install-hooks         # points core.hooksPath at .githooks (pre-commit is a no-op)
# or, with the env already active:
make test OP=<op>
```

`OP=<op>` selects `operation/<op>/` and `MODULE=<module>` selects `module/<module>/`; the rest of the flow is identical for both. It (1) runs `<layer>/<unit>/gen/gen_<unit>.py` to emit `<layer>/<unit>/vec/<unit>.vec` from the Python model, (2) compiles `<layer>/<unit>/rtl/*.v` + the testbench with `iverilog`, (3) simulates with `vvp` from inside the unit folder. The pass/fail decision is made by two greps over `<layer>/<unit>/build/sim.log`, not by the exit status of the `vvp` command. The testbench prints `PASS ...` only if every vector matches, and the Makefile requires a `^PASS` line; a `vvp` that dies, is absent, or prints nothing leaves a log without one and turns the target red. The Makefile also rejects a log carrying a `^ERROR` line, because `vvp` exits zero after a runtime error such as an unreadable `$readmemb` ROM. The `vvp | tee build/sim.log` pipe reports `tee`'s status, so nothing may rely on that pipe's exit code: the greps are the whole gate, and deleting either one removes a class of silent pass. `make test` runs `make guards` before the unit flow, so every target is a gate on the elaboration guards as well.

Every unit, operation or module, compiles with its own `rtl/*.v` plus every `operation/*/rtl` and `module/*/rtl` directory passed to `iverilog` as a `-y` library, so an instantiated circuit is pulled in from the file whose name equals it. An operation that reuses another operation's circuit instantiates that standalone module (e.g. `div_iscb_bipolar` instantiates `signabs`, `bi2uni`, and `uni2bi`). `make equiv` walks inlined copies only through the `PAIRS` table of `test_equiv.py`, currently seventeen pairs over fifteen host modules. An inlined copy is equiv-covered only when a row in that table names it, and the table is the only list of which inlined copies are equiv-covered: a copy with no row has no equiv coverage. Covered copies include, as examples, `lt_rc`'s `sync_skewed` datapath and `div_iscb_unipolar`'s `sync_skewed` and `div_cordiv` datapaths. A new inlined copy of a standalone operation's datapath gets its own row. `-y` resolves a name to the first matching file along that path, so basename uniqueness across both trees is what keeps the resolved file the intended one; `sweep_floor.py` fails on any `.v` basename appearing in more than one `rtl/` directory.

## Flow wiring

How the tree implements the rules above:

- **Variant selection.** Each polarity is its own module (`mul_gaines_unipolar` = AND, `mul_gaines_bipolar` = XNOR), so the generator picks a module by string-matching the PyTorch config (`f"{op}_{config['polarity']}"`) with no abbreviation table.
- **Sizing parameter flow.** `operation/<op>/gen/gen_<op>.py` reads the sizing config **once** (mirroring the op's `test_<op>.py` config), builds the Python model from it, generates the vectors from that model, and emits `operation/<op>/vec/<op>_params.vh` with one `` `define GEN_<PARAM> <value> `` per sizing field (`depth`, `width`, `scale`, `entry`, ...). `operation/<op>/tb/<op>_tb.v` does `` `include "<op>/vec/<op>_params.vh" `` (iverilog resolves it through `-Ioperation` from the compile cwd, `imp/`) and instantiates the DUT with the override `<op> #(.PARAM(`GEN_PARAM)) dut (...)`, so the *verified* hardware is always the model's configuration and the sim and the RTL cannot drift. Reading that config smaller than the op's own test, so a clamp lands inside a vector row, is licensed by module-layer rule 9 on the terms stated there.
- **Reset flow.** A unit whose RTL holds state carries `i_clk` and `i_rst_n`: its generator calls `model.reset()` and its testbench pulses `i_rst_n` low before driving, so the co-sim compares from the **first post-reset cycle** and a reset state that disagrees with the model fails on the opening vectors. `reset()` is the source of truth for that state: read it and reload whatever value it sets (e.g. `shiftreg` reloads the alternating `reg[i] = i % 2` pattern, not all-zeros). A unit whose RTL is pure `assign` carries neither port, and its generator's post-`reset()` replay checks the Python encoders the vectors are drawn from rather than an RTL reset. Which form a given unit takes is read off its own testbench, which either pulses `i_rst_n` or has no reset port at all.

## Review tiers

Review depth follows what a change touches. A change to RTL logic, a testbench, a generator, or `mapping.yaml` takes the full review: the reviewer requires the verification-step evidence above, the branch oracle behind any claim that an arm is dead, and the focused co-simulations (`make test OP=<op>` / `make test MODULE=<module>`) of every unit the change touches. A change confined to an RTL header banner, a comment, or prose in this file takes a one-pass spot check, with a verdict of at most three sentences.

Mechanical work is verified by command rather than by re-derivation. Work is mechanical when its correctness is establishable from the command output alone, without reading the diff: renames, formatting passes, and moves or anchor updates qualify. Work whose correctness depends on what the changed code means is not mechanical, however small the diff. The test is whether the attached command can fail in a way that proves the change wrong. For a rename, a port or parameter formatting pass, or a file move, the author attaches the command output that covers the claim: an `rg --hidden` sweep, or `make test OP=<op>` / `make test MODULE=<module>` for every unit touched, `make sweep` being an explicitly-called option rather than the default multi-unit evidence. The reviewer confirms the commands ran and that their scope covers the claim, spot-checks the results (a zero-hit sweep proves the old name is gone, not that the new name is right), and does not repeat the change by hand.
