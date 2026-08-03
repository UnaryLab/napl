# NAPL hardware rules

This file is the canonical design and verification policy for `src/napl/imp/`, plus its directory layout and commands. Where `src/napl/sim/operation/` is the *functional* Python model, the `src/napl/imp/` tree is its *hardware* counterpart: Verilog-2001 implementations of the NAPL stochastic-computing operations, one synthesizable module per concrete operation variant, each verified against the Python model with golden-vector co-simulation. Policy comes first; the layout and flow mechanics follow.

## Design contract

General Verilog style (Verilog-2001, `i_`/`o_` port prefixes, `i_clk`/active-low `i_rst_n`, one module per file with the file name equal to the module name, lint-clean) follows the unarylab-research plugin's verilog-rules hook, which injects the full rules when a Verilog file is edited. Separate every module, function, or task declaration from adjacent declarations or code in the same scope with exactly two blank lines.

1. **Variants and names.** Use `_unipolar` and `_bipolar` modules whenever polarity changes the circuit; do not select polarity-dependent logic with a parameter. An operation whose circuit is identical across supported polarities uses the bare operation name. The operation folder name and RTL module base name equal the `sim/operation` module name verbatim; the RTL module is `<op>` or `<op>_<polarity>`. A non-polarity Python configuration field that selects between circuits, such as `scaled`, is an integer Verilog parameter fixed at elaboration, never a runtime input. Distinct operations or port sets are separate operations with their own folders (e.g. `mul_and` vs `mul_csg`).
2. **Sizing parameters.** Sizing fields such as depth or width remain Verilog parameters derived from the Python configuration. Declare each with a default documenting the configuration it was generated for, and derive every bus width and internal constant from the parameters so the module is correct at any size.
3. **Spikes and cycles.** Spikes are 1-bit wires, one spike per stream per clock. Each RTL port name is its `i_` or `o_` prefix plus the corresponding Python `forward()` argument name verbatim, with no aliases. Each module is one scalar circuit per operation variant; vectorization across lanes is replication handled by higher-level blocks, not baked into the operation. One Python `forward()` timestep corresponds to one `posedge i_clk`.
4. **Reset.** Every register must return to the exact post-`reset()` state of the Python model when `i_rst_n` is asserted low. The correct state is not necessarily zero.
5. **Timing.** Record verified input-to-output latency in the Python operation as `self.hw = hw_params(pp_delay=...)`: zero for a combinational path and one cycle per register stage. If latency depends on a sizing field, derive both RTL latency and `pp_delay` from that field (e.g. `shiftreg`'s `pp_delay == config['depth'] == DEPTH`).

Per-corner STA data belongs in `self.hw.timing`, keyed by the MCMM scenario `node/process/voltage/temp/rc/mode`. A combinational operation records its full through-delay in `cp_delay` with zero `ir_delay` and `or_delay`; a registered operation records all three timing components.

6. **RTL coverage.** Every operation with `streaming = True` requires a verified RTL counterpart; operations with `streaming = False` (single-shot binary-domain: `relu_hub`, `sigmoid_hub`, `tanh_hub`, `round_fxp`) have none. Placeholder classes with no implementation (`inhibit`) carry no RTL obligation until implemented.
7. **Input handling.** Every streaming operation's RTL consumes its `i_*` data inputs combinationally in the arrival cycle: registers hold internal state only and never re-register a raw input before its first use. For delay and storage elements whose defined function is capturing the input (`dff`, `shiftreg`, `jkff`, `square_dff`'s internal delay), that capture is the first use.

## Per-operation structure

Each concrete operation lives in `operation/<op>/` with `rtl/`, `tb/`, `gen/`, `vec/`, and `build/` subdirectories. The testbench is `<op>_tb.v`; the generator is `gen_<op>.py`.

The generator builds the NAPL Python operation from one configuration, emits expected outputs from that model, and writes sizing values to `vec/<op>_params.vh`. The testbench includes that header and applies the same values to the RTL parameters. Do not hardcode a second copy of sizing values in the testbench or RTL.

`vec/*.vec`, `vec/*_params.vh`, and `build/` are generated and must remain untracked.

## Verification gates

Golden-vector co-simulation is the source of functional truth. Expected outputs must come from the NAPL Python model, never from a hand-written truth table. A single `<op>.vec` file carries the expected outputs for every variant; the testbench instantiates all of an operation's variant modules, must compare every output on the corresponding cycle, and the verification command must exit nonzero on any mismatch.

Every Python `test_*` function used by this verification flow starts with a concise one-line docstring that states the behavior it verifies.

For each RTL-backed change:

1. Run the operation's Python test under the rules in [RULE_SIM.md](RULE_SIM.md).
2. From `src/napl/imp/`, run `conda run -n napl make test OP=<op>` and require an exit status of zero and a full-match `PASS`.
3. For a stateful operation, verify the first post-reset cycle and a reset asserted after state has changed, then replay the same inputs.
4. Verify the measured cycle latency against `self.hw.pp_delay`, including every supported sizing configuration used by the test.
5. Report the command, configuration and sizing parameters, vector count, reset cases, observed latency, expected `pp_delay`, and exit status.

## Directory layout

The tree contains 36 implemented op folders under `operation/`, plus the generic `Makefile` and the shared golden-vector helper `operation/_gen_common.py` (`encode_value`, `pair_streams`, ...). Implementations are added one op folder at a time, either by hand following this file or via the `napl-port-unarysim` workflow (which generates each op's RTL + testbench + generator and verifies it against the Python model). `mul_and` (unipolar = AND, bipolar = XNOR) is the canonical example referenced throughout.

Each operation is self-contained in its own folder:

```
imp/
├── Makefile                  # make test OP=<op>
└── operation/
    ├── _gen_common.py        # shared golden-vector helpers (encode_value, pair_streams, ...)
    └── <op>/                 # one folder per operation, e.g. mul_and/
        ├── rtl/              # hand-written RTL modules, one per variant
        │   ├── <op>_unipolar.v
        │   └── <op>_bipolar.v
        ├── tb/               # <op>_tb.v -- self-checking testbench
        ├── gen/              # gen_<op>.py -- golden vectors + param header FROM the napl Python model
        ├── vec/              # generated <op>.vec and <op>_params.vh (gitignored)
        └── build/            # compiled sim + log (auto-created, gitignored)
```

## Running a test

Requires the Icarus Verilog toolchain (`iverilog` / `vvp`) and the `napl` conda env (the vector generator does `import napl`):

```bash
conda run -n napl make test OP=<op>
# or, with the env already active:
make test OP=<op>
```

This (1) runs `operation/<op>/gen/gen_<op>.py` to emit `operation/<op>/vec/<op>.vec` from the Python model, (2) compiles `operation/<op>/rtl/*.v` + the testbench with `iverilog`, (3) simulates with `vvp` from inside the op folder. The testbench prints `PASS ...` only if every vector matches; the Makefile greps for it, so a mismatch makes `make` exit non-zero.

Compilation pulls in only `operation/<op>/rtl/*.v`, so an op is self-contained. An op that reuses another op's primitive embeds that logic in its own `rtl/` (e.g. `lt_rc` embeds the `sync_skewed` datapath).

## Flow wiring

How the tree implements the rules above:

- **Variant selection.** Each polarity is its own module (`mul_and_unipolar` = AND, `mul_and_bipolar` = XNOR), so the generator picks a module by string-matching the PyTorch config (`f"{op}_{config['polarity']}"`) with no abbreviation table.
- **Sizing parameter flow.** `operation/<op>/gen/gen_<op>.py` reads the sizing config **once** (mirroring the op's `test_<op>.py` config), builds the Python model from it, generates the vectors from that model, and emits `operation/<op>/vec/<op>_params.vh` with one `` `define GEN_<PARAM> <value> `` per sizing field (`depth`, `width`, `scale`, `entry`, ...). `operation/<op>/tb/<op>_tb.v` does `` `include "<op>/vec/<op>_params.vh" `` (iverilog resolves it through `-Ioperation` from the compile cwd, `imp/`) and instantiates the DUT with the override `<op> #(.PARAM(`GEN_PARAM)) dut (...)`, so the *verified* hardware is always the model's configuration and the sim and the RTL cannot drift.
- **Reset flow.** The golden-vector generator calls `model.reset()` and the testbench pulses `i_rst_n` low before driving, so the co-sim compares from the **first post-reset cycle**; a reset state that disagrees with the model fails on the opening vectors. `reset()` is the source of truth: read it and reload whatever value it sets (e.g. `shiftreg` reloads the alternating `reg[i] = i % 2` pattern, not all-zeros). Combinational ops are pure `assign`, no clock (e.g. `mul_and_*`); stateful ops (`dff`, `shiftreg`, `add_any`, `div_cordiv`, `sqrt_emit`, ...) add `i_clk` and `i_rst_n`.
