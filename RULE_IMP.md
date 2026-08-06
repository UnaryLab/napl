# NAPL hardware rules

This file is the canonical design and verification policy for `src/napl/imp/`, plus its directory layout and commands. Where `src/napl/sim/operation/` is the *functional* Python model, the `src/napl/imp/` tree is its *hardware* counterpart: Verilog-2001 implementations of the NAPL stochastic-computing operations, one synthesizable module per concrete operation variant, each verified against the Python model with golden-vector co-simulation. `src/napl/imp/module/` extends the same flow to `sim/module` classes, whose lane-replicated RTL composes those operation circuits. Policy comes first; the layout and flow mechanics follow.

## Design contract

General Verilog style (Verilog-2001, `i_`/`o_` port prefixes, `i_clk`/active-low `i_rst_n`, one module per file with the file name equal to the module name, lint-clean) follows the unarylab-research plugin's verilog-rules hook, which injects the full rules when a Verilog file is edited. Separate every module, function, or task declaration from adjacent declarations or code in the same scope with exactly two blank lines.

1. **Variants and names.** Use `_unipolar` and `_bipolar` modules whenever polarity changes the circuit; do not select polarity-dependent logic with a parameter. An operation whose circuit is identical across supported polarities uses the bare operation name. The operation folder name and RTL module base name equal the `sim/operation` module name verbatim; the RTL module is `<op>` or `<op>_<polarity>`. A non-polarity Python configuration field that selects between circuits, such as `scaled`, is an integer Verilog parameter fixed at elaboration, never a runtime input. Distinct operations or port sets are separate operations with their own folders (e.g. `mul_gaines` vs `mul_ugemm`).
2. **Sizing parameters.** Sizing fields such as depth or width remain Verilog parameters derived from the Python configuration. Declare each with a default documenting the configuration it was generated for, and derive every bus width and internal constant from the parameters so the module is correct at any size.
3. **Spikes and cycles.** Spikes are 1-bit wires, one spike per stream per clock. Each RTL port name is its `i_` or `o_` prefix plus the corresponding Python `forward()` argument name verbatim, with no aliases. Each module is one scalar circuit per operation variant; vectorization across lanes is replication handled by higher-level blocks, not baked into the operation. One Python `forward()` timestep corresponds to one `posedge i_clk`.
4. **Reset.** Every register must return to the exact post-`reset()` state of the Python model when `i_rst_n` is asserted low. The correct state is not necessarily zero.
5. **Timing.** Record verified input-to-output latency in the Python operation as `self.hw = hw_params(pp_delay=...)`: zero for a combinational path and one cycle per register stage. If latency depends on a sizing field, derive both RTL latency and `pp_delay` from that field (e.g. `shiftreg`'s `pp_delay == config['depth'] == DEPTH`).

Per-corner STA data belongs in `self.hw.timing`, keyed by the MCMM scenario `node/process/voltage/temp/rc/mode`. A combinational operation records its full through-delay in `cp_delay` with zero `ir_delay` and `or_delay`; a registered operation records all three timing components.

6. **RTL coverage.** Every operation with `streaming = True` requires a verified RTL counterpart; operations with `streaming = False` (single-shot binary-domain: `relu_hub`, `sigmoid_hub`, `tanh_hub`, `round_fxp`) have none.
7. **Input handling.** Every streaming operation's RTL consumes its `i_*` data inputs combinationally in the arrival cycle: registers hold internal state only and never re-register a raw input before its first use. For delay and storage elements whose defined function is capturing the input (`dff`, `shiftreg`, `jkff`, `square_dff`'s internal delay), that capture is the first use.
8. **Explicit imports.** Python generator and testbench scripts under `src/napl/imp/` use explicit imports and do not use `from x import *`. Package `__init__.py` re-exports may use star imports.
9. **Seeded sequences.** A configuration a generator builds a model or encoder from sets `seed` whenever `generator` is `sys`. An unseeded `sys` sequence is drawn fresh per instance, so the emitted ROM would not match the sequence a separately constructed model draws, and the co-simulation cannot reveal the mismatch because it builds a single model. The check lives in `encode.__init__` (`sim/operation/encode.py`), which applies `require_seeded_sys()` to its own configuration, so every encoder a generator builds is covered whichever way it builds it and no generator can skip the check by omitting a call. `gen_num_seq()` stays permissive, because an unseeded `sys` sequence is a valid simulation-side draw, so a generator still calls `require_seeded_sys()` itself for a model configuration that never reaches an encoder. `operation/_gen_common.py` re-exports the function for those calls.

## Per-operation structure

Each concrete operation lives in `operation/<op>/` with `rtl/`, `tb/`, `gen/`, `vec/`, and `build/` subdirectories. The testbench is `<op>_tb.v`; the generator is `gen_<op>.py`.

The generator builds the NAPL Python operation from one configuration, emits expected outputs from that model, and writes sizing values to `vec/<op>_params.vh`. The testbench includes that header and applies the same values to the RTL parameters. Do not hardcode a second copy of sizing values in the testbench or RTL.

`vec/*.vec`, `vec/*_params.vh`, and `build/` are generated and must remain untracked.

## Verification gates

Golden-vector co-simulation is the source of functional truth. Expected outputs must come from the NAPL Python model, never from a hand-written truth table. A single `<op>.vec` file carries the expected outputs for every variant; the testbench instantiates all of an operation's variant modules, must compare every output on the corresponding cycle, and the verification command must exit nonzero on any mismatch.

Every Python `test_*` function used by this verification flow starts with a concise one-line docstring that states the behavior it verifies.

RTL co-simulation uses the fidelity-scale input shape from the streaming suite's `make_values`; the `make_performance_values` workload is for Python-side timing only and is not required for RTL vectors or co-simulation.

For each RTL-backed change:

1. Run the operation's Python test under the rules in [RULE_SIM.md](RULE_SIM.md).
2. From `src/napl/imp/`, run `conda run -n napl make test OP=<op>` for an operation or `conda run -n napl make test MODULE=<module>` for a module, and require an exit status of zero and a full-match `PASS`. The two selectors are exclusive; giving both is an error.
3. For a stateful operation, verify the first post-reset cycle and a reset asserted after state has changed, then replay the same inputs.
4. Verify the measured cycle latency against `self.hw.pp_delay`, including every supported sizing configuration used by the test.
5. Report the command, configuration and sizing parameters, vector count, reset cases, observed latency, expected `pp_delay`, and exit status.

## Module layer

`module/<name>/` holds the hardware counterpart of a `sim/module` class and mirrors the operation layout exactly: `rtl/`, `tb/`, `gen/`, `vec/`, and `build/`, with testbench `<name>_tb.v` and generator `gen_<name>.py`. The design contract and verification gates above apply unchanged; the following are the module-layer specifics.

1. **Composition.** Module RTL composes operation-layer scalar circuits. A module instantiates the `operation/*/rtl/` module whose function it needs instead of restating that logic, so the operation layer stays the single implementation of every scalar circuit.
2. **Vectorization.** A module covers many lanes: one lane per tensor position the Python class produces. Lanes are `generate`-`for` replications of the scalar circuit, and the lane count is a Verilog parameter (`LANES`) derived from the Python input shape and geometry. Lane `l` occupies the low-to-high slice `[l*<lane width> +: <lane width>]` of each vector port.
3. **Ports.** Each port name is its `i_` or `o_` prefix plus the corresponding Python `forward()` argument name verbatim, as in rule 3 of the design contract. Spikes remain 1-bit per lane per clock, so a port carrying `LANES` lanes is a packed vector of them.

    A parallel-counter module (`conv_pc`, `linear_pc`) is the exception: its lane produces a count, not a spike, so its output port is a **count bus** of `LANES` fields of `COUNT_W` bits, lane `l` at `[l*COUNT_W +: COUNT_W]`. `COUNT_W` is a parameter rather than a body `localparam` because a Verilog-2001 port width cannot read one, so an elaboration guard checks it equals `clog2(ENTRY + 1)`. `mapping.yaml` derives `COUNT_W` from `ENTRY` and therefore cannot violate that equality, which is why it carries no matching `requires` condition; the guard covers a hand-instantiated module.
4. **Golden vectors.** Expected outputs come from the Python module model in `sim/module/`, driven by the encoder configuration and the fidelity-scale input shape of the class's `test_<name>.py`. The generator flattens the model's per-timestep input and output tensors into the lane order the RTL ports use.
5. **Timing.** One Python `forward()` timestep corresponds to one `posedge i_clk`, the same as the operation layer, and the testbench checks the observed latency against the class's `self.hw.pp_delay`.

6. **Elaboration guards.** A sizing restriction the RTL cannot check at runtime is enforced at elaboration by instantiating an undefined module inside a `generate` `if`, so the build fails with that module's name. The guard module is named `ERROR_<base>_<restriction>`, where `<base>` is the polarity-stripped operation or module base name, so every polarity variant of one restriction reports the same name. Guards are elaboration-time constant folding and carry zero area, so they are exempt from the "no excess sanity-checking logic" rule, which targets runtime logic. Write the guard condition so it does not overflow 32-bit integer arithmetic: compare widths (`WIDTH - 1 < clog2(ENTRY + 1)`) rather than the values themselves (`2 ** (WIDTH - 1) <= ENTRY`).

    The guarded modules are `avgpool2d` (`DIVISOR >= KERNEL_AREA`), `linear_*`, `linear_ugemm_*`, `conv_*` and `conv_ugemm_*` (`2 ** (WIDTH - 1) > ENTRY`, plus `LANES` equal to the output positions the convolution geometry produces for `conv_*` and `conv_ugemm_*`), `linear_pc_*` and `conv_pc_*` (`COUNT_W == clog2(ENTRY + 1)`, plus the same `LANES` restriction for `conv_pc_*`), `add_any_*` (`2 ** WIDTH + 3 * ENTRY + 1 <= 2 ** 31 - 1`, the range of the 32-bit signed elaboration constants, which caps `WIDTH` at 30 on its own), and `div_cordiv` (`DEPTH == 2 ** WIDTH`, which also rules out an empty buffer). `linear_ugemm_*` and `conv_ugemm_*` compose `add_any_*`, so they carry that module's `WIDTH <= 30` cap as a `requires` condition too. Every such restriction is also stated in the RTL header banner and mirrored as a `requires` condition in `mapping.yaml`, so translation rejects the configuration before elaboration does, and enrolled as a row in `imp/test_guards.py`, run by `make guards` and by every `make test` target: each row elaborates a violating parameter set and requires `iverilog` to fail with the guard's name, plus a legal set right below the boundary that must still elaborate. A guard that is deleted or written backwards fails that test, and fails every unit test with it.

    A guard term whose violating parameter set cannot be elaborated has no row: `add_any_*`'s `ENTRY` term needs a bus of hundreds of millions of lanes, so `mapping.yaml`'s `requires`, evaluated in Python, is the only check on it.

7. **Golden-column widths.** `%b` zero-extends a short vector silently, so a testbench that scans a golden column straight into a sized `reg` passes at the wrong lane count. Scan each vector column as `%s`, check its character count against the port width, and convert the characters to bits (`token_len`, `token_bits`, and `check_width` in `conv_tb.v` are the pattern to copy). A testbench copied from an existing module testbench inherits these three helpers with the rest of the pattern.

    A count-bus column is that one pattern at a different width: the column is the lane counts packed exactly as the port packs them, `LANES * COUNT_W` characters highest bit index first, and `check_width` compares against `LANES * COUNT_W` instead of `LANES`. That width assertion is what ties `COUNT_W` to the hardware, since the generator and the RTL would otherwise agree on a wrong count width.

    The token registers are one character wider than the widest golden column (`MAX_CHARS = <widest column> + 1`). `$fscanf` with `%s` truncates to the token width, so a token sized exactly to the widest column saturates at the expected length and an over-long column of that width passes the check. The spare character makes an over-long column read back long and fail.

8. **Vector count.** The generator emits its row count as `` `define GEN_VECTORS ``, and the testbench requires the number of rows it consumed to equal it, so a vec file that lost or gained rows fails instead of passing on the rows it still holds. The row loop ends on the first `$fscanf` that does not return every column, rather than looping on `$feof`, which spins when a scan stops consuming.

The implemented modules are `avgpool2d`, `linear`, `linear_ugemm`, `conv`, `linear_pc`, `conv_pc`, and `conv_ugemm` (all but the first with one RTL variant per polarity). `conv`, `conv_pc`, and `conv_ugemm` wire the im2col patch in hardware, so their input port carries the NCHW spike tensor. In `conv` and `conv_pc` a tap outside the input reads `i_pad_bits`, the pad spike the Python model encodes on its own sobol dimension.

`conv_ugemm` generates its weight, bias, and pad streams inside the RTL, matching the Python class's `internal_encode = True`: the weight and bias ports carry held fixed-point codes rather than spikes, and there is no pad port. Its bipolar pad is the model's alternating 0/1 stream, built from a `jkff` held at J = K = 1, whose reset state is the model's opening `0`; its unipolar pad is a zero spike. A padded tap advances the sequence index of the `mul_ugemm_*` cell it drives, so it is never a don't-care. The Python model shares one sequence-index pair per (spatial output position, patch tap) across the output channels, because the patch spike broadcasts over them; each RTL lane holds its own pair, and every copy is advanced by the same patch spike, so the copies stay equal and the outputs are bit-exact with the shared-index model.

`linear_pc` and `conv_pc` are `linear` and `conv` with the `add_any_*` stage removed and the popcount it reduces published on a count bus. They hold no accumulator, so they are combinational and carry no `i_clk` or `i_rst_n`, and their testbenches apply and compare each row on the spot with no clock edge. Their generators still record the three-sequence replay, which checks the Python encoders the vectors are drawn from rather than an RTL reset. The popcount is written inside the module because the operation layer holds no standalone popcount circuit: `add_any` bundles one with the scaled accumulator these modules do not have.

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

`WIDTH` and `add_ugemm_*`'s `ACC_WIDTH` are checked by nothing. They are pass-through or derived accumulator widths, and the vectors never drive the accumulator near its bound, so a value that is too small changes no compared output and a value that is too large is invisible. Neither the generator assertion, the testbench, nor the elaboration guards close that gap; only the `requires` conditions bound them, and only at their overflow limit. A stimulus row that saturates the accumulator, added to a future vector set, would make a wrong width observable.

The `napl-gen-rtl` skill generates and verifies operation folders only. Module folders are written by hand under this file.

## Directory layout

The tree contains 43 implemented op folders under `operation/` and seven module folders under `module/`, plus the generic `Makefile`, the shared golden-vector helper `operation/_gen_common.py` (`encode_value`, `pair_streams`, ...), and `test_guards.py`, the elaboration-guard test. Implementations are added one op folder at a time, either by hand following this file or via the `napl-port-unarysim` workflow (which generates each op's RTL + testbench + generator and verifies it against the Python model). `mul_gaines` (unipolar = AND, bipolar = XNOR) is the canonical example referenced throughout.

Each operation is self-contained in its own folder:

```
imp/
├── Makefile                  # make test OP=<op> | make test MODULE=<module>
├── test_guards.py            # make guards -- every elaboration guard, violating and legal
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
    └── <module>/             # one folder per module, e.g. avgpool2d/
        ├── rtl/              # lane-replicated RTL composed of operation-layer circuits
        ├── tb/               # <module>_tb.v -- self-checking testbench
        ├── gen/              # gen_<module>.py -- golden vectors + param header FROM the napl Python model
        ├── vec/              # generated <module>.vec and <module>_params.vh (gitignored)
        └── build/            # compiled sim + log (auto-created, gitignored)
```

## Running a test

Requires the Icarus Verilog toolchain (`iverilog` / `vvp`) and the `napl` conda env (the vector generator does `import napl`):

```bash
conda run -n napl make test OP=<op>          # operation layer
conda run -n napl make test MODULE=<module>  # module layer
conda run -n napl make guards                # every elaboration guard, all units (also run by make test)
# or, with the env already active:
make test OP=<op>
```

`OP=<op>` selects `operation/<op>/` and `MODULE=<module>` selects `module/<module>/`; the rest of the flow is identical for both. It (1) runs `<layer>/<unit>/gen/gen_<unit>.py` to emit `<layer>/<unit>/vec/<unit>.vec` from the Python model, (2) compiles `<layer>/<unit>/rtl/*.v` + the testbench with `iverilog`, (3) simulates with `vvp` from inside the unit folder. The testbench prints `PASS ...` only if every vector matches; the Makefile greps for it, so a mismatch makes `make` exit non-zero. It also rejects a log carrying an `ERROR` line, because `vvp` exits zero after a runtime error such as an unreadable `$readmemb` ROM. `make test` runs `make guards` before the unit flow, so every target is a gate on the elaboration guards as well.

An operation compiles from `operation/<op>/rtl/*.v` alone, so an op is self-contained. An op that reuses another op's primitive embeds that logic in its own `rtl/` (e.g. `lt_rc` embeds the `sync_skewed` datapath). A module compiles with every `operation/*/rtl` directory passed to `iverilog` as a `-y` library, so an instantiated operation module is pulled in from the file whose name equals it.

## Flow wiring

How the tree implements the rules above:

- **Variant selection.** Each polarity is its own module (`mul_gaines_unipolar` = AND, `mul_gaines_bipolar` = XNOR), so the generator picks a module by string-matching the PyTorch config (`f"{op}_{config['polarity']}"`) with no abbreviation table.
- **Sizing parameter flow.** `operation/<op>/gen/gen_<op>.py` reads the sizing config **once** (mirroring the op's `test_<op>.py` config), builds the Python model from it, generates the vectors from that model, and emits `operation/<op>/vec/<op>_params.vh` with one `` `define GEN_<PARAM> <value> `` per sizing field (`depth`, `width`, `scale`, `entry`, ...). `operation/<op>/tb/<op>_tb.v` does `` `include "<op>/vec/<op>_params.vh" `` (iverilog resolves it through `-Ioperation` from the compile cwd, `imp/`) and instantiates the DUT with the override `<op> #(.PARAM(`GEN_PARAM)) dut (...)`, so the *verified* hardware is always the model's configuration and the sim and the RTL cannot drift.
- **Reset flow.** The golden-vector generator calls `model.reset()` and the testbench pulses `i_rst_n` low before driving, so the co-sim compares from the **first post-reset cycle**; a reset state that disagrees with the model fails on the opening vectors. `reset()` is the source of truth: read it and reload whatever value it sets (e.g. `shiftreg` reloads the alternating `reg[i] = i % 2` pattern, not all-zeros). Combinational ops are pure `assign`, no clock (e.g. `mul_gaines_*`); stateful ops (`dff`, `shiftreg`, `add_any`, `div_cordiv`, `sqrt_emit`, ...) add `i_clk` and `i_rst_n`.
