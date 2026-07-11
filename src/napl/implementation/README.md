# NAPL RTL implementation

Verilog (Verilog-2001) implementations of the NAPL stochastic-computing
operations. Where `src/napl/operation/` is the *functional* Python model, this
tree is its *hardware* counterpart: one synthesizable module per concrete
operation variant, each verified against the Python model with golden-vector
co-simulation.

> **Status:** this tree contains 27 implemented op folders, plus the generic
> `Makefile`, the shared golden-vector helper `_gen_common.py`, and this spec.
> Implementations are added one op folder at a time, either by hand following
> the conventions below or via the `napl-port-unarysim` workflow (which
> generates each op's RTL + testbench + generator and verifies it against the
> Python model). `mul_and` (unipolar = AND, bipolar = XNOR) is the canonical
> example referenced throughout.

## Layout

Each operation is self-contained in its own folder:

```
implementation/
├── Makefile                  # make test OP=<op>
├── README.md
├── _gen_common.py            # shared golden-vector helpers (encode_value, pair_streams, ...)
└── <op>/                     # one folder per operation, e.g. mul_and/
    ├── rtl/                  # hand-written RTL modules, one per variant
    │   ├── <op>_unipolar.v
    │   └── <op>_bipolar.v
    ├── tb/                   # <op>_tb.v -- self-checking testbench
    ├── gen/                  # gen_<op>.py -- golden vectors + param header FROM the napl Python model
    ├── vec/                  # generated <op>.vec and <op>_params.vh (gitignored)
    └── build/                # compiled sim + log (auto-created, gitignored)
```

## Running a test

Requires the Icarus Verilog toolchain (`iverilog` / `vvp`) and the `napl` conda
env (the vector generator does `import napl`):

```bash
conda run -n napl make test OP=<op>
# or, with the env already active:
make test OP=<op>
```

This (1) runs `<op>/gen/gen_<op>.py` to emit `<op>/vec/<op>.vec` from the Python
model, (2) compiles `<op>/rtl/*.v` + the testbench with `iverilog`, (3)
simulates with `vvp` from inside the op folder. The testbench prints `PASS ...`
only if every vector matches; the Makefile greps for it, so a mismatch makes
`make` exit non-zero.

Compilation pulls in only `<op>/rtl/*.v`, so an op is self-contained. An op that
reuses another op's primitive (e.g. `compare` building on `sync_skewed`) gets an
explicit dependency list when it is rolled out.

## Conventions

- **One module per concrete variant, no parameters for variant selection.**
  Each polarity is its own module (`mul_and_unipolar` = AND, `mul_and_bipolar` =
  XNOR), so the generator picks a module by string-matching the PyTorch config
  (`f"{op}_{config['polarity']}"`) with no abbreviation table.
- **Sizing params are inherited from the Python model, not hardcoded.** A variant
  is *selected* by module name (above); its *size* (`depth`, `width`, `scale`,
  `entry`, ...) is a Verilog `parameter` (UPPER_CASE) inherited from the model's
  config. This is orthogonal to variant selection: there is still one module per
  polarity, but its dimensions are not baked-in constants. The contract:
  - The RTL module declares each sizing field as a `parameter` with a default
    documenting the config it was generated for (e.g. `module shiftreg #(parameter
    integer DEPTH = 2) (...)`), and **derives every bus width and magic constant
    from those parameters** (`reg [DEPTH-1:0]`, `ACC_HI = 2**(WIDTH-1)-1`, ...) so
    the module is correct at any size, not just the default.
  - `gen/gen_<op>.py` reads the sizing config **once** (mirroring the op's
    `test_<op>.py` config), builds the Python model from it, generates the vectors
    from that model, and emits `vec/<op>_params.vh` with one `` `define GEN_<PARAM>
    <value> `` per sizing field. This file is the single source of truth, so the
    sim and the RTL cannot drift.
  - `tb/<op>_tb.v` does `` `include "<op>/vec/<op>_params.vh" `` (iverilog resolves
    the path relative to the compile cwd, `implementation/`) and instantiates the
    DUT with the override `<op> #(.PARAM(`GEN_PARAM)) dut (...)`, so the *verified*
    hardware is always the model's configuration.
- **Module naming.** The RTL module name (and its file name) is *exactly* the op
  name (which may itself contain underscores, e.g. `sqrt_traceiscb`), optionally
  followed by a **single** polarity postfix `_unipolar` or `_bipolar` when the op
  has distinct polarity variants. **No other postfix** (no mode, config, version,
  or width tags). An op with no polarity split uses the bare op name. Distinct
  operations / port sets are distinct ops with their own folders (`mul_and` vs
  `mul_csg`).
- **Per-op folders:** an op's RTL, testbench, generator and vectors all live
  under `<op>/` in `rtl/`, `tb/`, `gen/`, `vec/`. The testbench is `<op>_tb.v`;
  the generator is `gen_<op>.py`.
- **Spikes** are 1-bit `wire`s; one spike per stream per clock. Per the global
  Verilog rules (auto-injected by the `verilog-rules.sh` hook on any `.v` edit),
  inputs are prefixed `i_` and outputs `o_`, so the
  ports trace to the Python `forward()` arguments as `i_in_0`, `i_in_1`, `o_out`.
  One scalar circuit per (op, variant); vectorization across lanes is replication
  handled by higher-level blocks (linear/conv), not baked in here.
- **Verification** is golden-vector co-simulation: expected outputs always come
  from the napl Python model, never a hand-written truth table. One `<op>.vec`
  file carries the expected output for every variant; the testbench instantiates
  all of an op's variant modules and checks each.
- **Pipeline delay → `self.hw.pp_delay`.** An op's RTL latency in clock cycles
  from an input to its corresponding output (0 for purely combinational, 1 per
  register stage) is recorded back in the Python class's hardware contract
  `self.hw = hw_params(pp_delay=...)` so the functional model and the hardware
  agree on timing. When the
  delay itself depends on a sizing param it follows from it, e.g. `shiftreg`'s
  `pp_delay == config['depth'] == DEPTH`.

### Combinational vs. clocked

Per the global Verilog rules (auto-injected by the `verilog-rules.sh` hook): the
clock is always `i_clk` and the reset is active-low `i_rst_n`. Combinational ops are pure `assign`, no clock (e.g.
`mul_and_*`). **Stateful** operations (`dff`, `shiftreg`, `add_any`, `div`,
`sqrt`, …) add `input wire i_clk` (posedge) and an active-low `i_rst_n` that maps
to the Python `reset()`. The mapping is: one Python `forward()` timestep == one
`posedge i_clk`, and asserting `i_rst_n` low must bring every register to **the
exact post-`reset()` state of the Python model**, which is *not always zero*.
`reset()` is the source of truth: read it and reload whatever value it sets (e.g.
`shiftreg` reloads the alternating `reg[i] = i % 2` pattern, not all-zeros). The
golden-vector generator calls `model.reset()` and the testbench pulses `i_rst_n`
low before driving, so the co-sim compares from the **first post-reset cycle**;
a reset state that disagrees with the model fails on the opening vectors.
