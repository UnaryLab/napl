# NAPL RTL implementation

Verilog (Verilog-2001) implementations of the NAPL stochastic-computing
operations. Where `src/napl/operation/` is the *functional* Python model, this
tree is its *hardware* counterpart: one synthesizable module per concrete
operation variant, each verified against the Python model with golden-vector
co-simulation.

Each operation is self-contained in its own folder. `mul_and` is the worked
example / template; the rest of the operations follow the same conventions.

## Layout

```
implementation/
├── Makefile                  # make test OP=<op>
├── README.md
└── <op>/                     # one folder per operation, e.g. mul_and/
    ├── rtl/                  # hand-written RTL modules, one per variant
    │   ├── <op>_unipolar.v
    │   └── <op>_bipolar.v
    ├── tb/                   # <op>_tb.v -- self-checking testbench
    ├── gen/                  # gen_<op>.py -- golden vectors FROM the napl Python model
    ├── vec/                  # generated <op>.vec (gitignored)
    └── build/                # compiled sim + log (auto-created, gitignored)
```

## Running a test

Requires the Icarus Verilog toolchain (`iverilog` / `vvp`) and the `napl` conda
env (the vector generator does `import napl`):

```bash
conda run -n napl make test OP=mul_and
# or, with the env already active:
make test OP=mul_and
```

This (1) runs `<op>/gen/gen_<op>.py` to emit `<op>/vec/<op>.vec` from the Python
model, (2) compiles `<op>/rtl/*.v` + the testbench with `iverilog`, (3)
simulates with `vvp` from inside the op folder. The testbench prints `PASS ...`
only if every vector matches; the Makefile greps for it, so a mismatch makes
`make` exit non-zero.

Compilation pulls in only `<op>/rtl/*.v`, so an op is self-contained. An op that
reuses another op's primitive (e.g. `compare` building on `sync_skewed`) will
get an explicit dependency list when we roll those out; `mul_and` needs none.

## Conventions

- **One module per concrete variant — no parameters for variant selection.**
  Each polarity is its own module (`mul_and_unipolar` = AND,
  `mul_and_bipolar` = XNOR). The variant suffix is *exactly* the Python `config`
  value it comes from (`config['polarity']` → `mul_and_<polarity>`), so the RTL
  generator picks a module by string-matching the PyTorch config
  (`f"{op}_{config['polarity']}"`) with no abbreviation table. Distinct
  operations / port sets are distinct ops with their own folders (`mul_and` vs
  `mul_csg`).
- **Per-op folders:** an op's RTL, testbench, generator and vectors all live
  under `<op>/` in `rtl/`, `tb/`, `gen/`, `vec/`. The RTL module/file name is the
  variant name; the testbench is `<op>_tb.v`; the generator is `gen_<op>.py`.
- **Spikes** are 1-bit `wire`s; one spike per stream per clock. Port names trace
  to the Python `forward()` arguments (`in_0`, `in_1`, `out`). One scalar circuit
  per (op, variant) — vectorization across lanes is replication handled by
  higher-level blocks (linear/conv), not baked in here.
- **Verification** is golden-vector co-simulation: expected outputs always come
  from the napl Python model, never a hand-written truth table. One `<op>.vec`
  file carries the expected output for every variant; the testbench instantiates
  all of an op's variant modules and checks each.

### Combinational vs. clocked

`mul_and_*` are **combinational** — pure `assign`, no clock. **Stateful**
operations (`dff`, `shiftreg`, `add_any`, `div`, `sqrt`, …) will add
`input wire clk` (posedge) and an active-low `rst_n` that maps to the Python
`reset()`. The mapping is: one Python `forward()` timestep == one `posedge clk`,
and asserting `rst_n` low clears the same state that `reset()` zeroes.
