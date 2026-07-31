# NAPL

A neuro-adaptive programming language for general-purpose neuromorphic computing.

## Overview

NAPL is UnaryLab's PyTorch framework for programmable spike processing. It runs per-timestep spike-stream simulation and single-shot binary-domain kernels, with verified Verilog counterparts for a subset of operations.

- [ARCHITECTURE.md](ARCHITECTURE.md): dataflow, execution models, package map, and the Python-to-RTL contract. Read it before editing execution semantics.
- [RULE_SIM.md](RULE_SIM.md) and [RULE_IMP.md](RULE_IMP.md): the simulation test policy and the hardware design/verification policy.
- [`src/napl/imp/`](src/napl/imp/): the Verilog counterpart of `napl.sim.operation`, with RTL co-simulation via `conda run -n napl make -C src/napl/imp test OP=<op>`.

## Requirements

- [Anaconda](https://www.anaconda.com/) (the environment is defined in `environment.yaml`).
- Icarus Verilog (`iverilog`/`vvp`) on `PATH`, only for RTL co-simulation.

## Installation

NAPL is not yet published on PyPI; install from source. This installs in editable mode: source edits take effect without reinstalling.

1. `git clone` [this repo](https://github.com/UnaryLab/napl) and `cd` into the repo directory.
2. `conda env create -f environment.yaml`
   - The `name: napl` in `environment.yaml` can be changed to a preferred one.
3. `conda activate napl`
4. `python3 -m pip install -e . --no-deps`
5. Check the install with `napl -h` in the command line, or `import napl` in Python.

The `napl` command line entry point currently prints a banner; it is a placeholder.

## Quick start

Every public class is importable at the top level, e.g. `from napl import linear, mul_and, accuracy`.

Run a single kernel test:

```sh
conda run -n napl python tests/operation/test_mul_and.py
```

Run the full suite, which walks `tests/` and runs every `test_*.py` as a standalone script:

```sh
conda run -n napl python tests/sweep_test.py
```

## Documentation

The Sphinx sources live under [`docs/`](docs/). Build the design documentation
and docstring-based API reference as static HTML with:

```sh
conda run -n napl make -C docs html
```

The output is written to `docs/_build/html/`.

## Reproducing results

[`examples/`](examples/) holds three end-to-end UnarySim app ports, each with its own README and `results/*.csv` outputs: `mlp` (MLP3 on MNIST), `ubrain` (uBrain BCI CNN+RNN), and `usystolic` (uSystolic convnet).

## Configuration

`environment.yaml` defines the conda environment. `src/napl/sim/base/global_config.yaml`
sets the global spike and non-spike PyTorch data types, and each kernel takes its
parameters as a `config` dict at construction.

## Citation

Not yet published.

## License

MIT, see [LICENSE](LICENSE).
