Architecture
============

NAPL is a PyTorch framework for programmable spike processing. It supports
per-timestep spike-stream simulation and single-shot binary-domain kernels.
The Python model defines functional behavior. A subset of spike operations has
verified Verilog-2001 counterparts.

The root `ARCHITECTURE.md
<https://github.com/UnaryLab/napl/blob/main/ARCHITECTURE.md>`_ is the canonical
architecture specification. This page provides a concise view for the HTML
documentation.

Dataflow
--------

The streaming path encodes a number or tensor into a one-bit spike stream,
processes one spike from each input per timestep, and decodes or measures the
result over time::

   number or tensor
          |
       encode
          |
   one-bit spike stream
          |
   operations or streaming layers
          |
   decode or metric

Rate and temporal encodings differ in spike order. Longer streams generally
improve numerical fidelity at the cost of latency.

Single-shot kernels process a complete tensor in one call. They use
quantization or hybrid unary-binary arithmetic without an explicit timestep
stream.

Package map
-----------

``src/napl/sim/base/``
   Shared lifecycle, timestep decorators, hardware metadata, and dtype
   configuration.

``src/napl/sim/module/``
   Encoders, decoders, neural layers, and recurrent cells.

``src/napl/sim/operation/``
   Arithmetic, comparison, activation, state, polarity-conversion, and stream
   synchronization primitives.

``src/napl/sim/metric/``
   Progressive stream observers and stability stream construction.

``src/napl/sim/algorithm/``
   Compositions of modules and operations, including an FFT butterfly.

``src/napl/sim/structure/``
   Placeholder boundaries for biological-neuron abstractions.

``src/napl/imp/``
   Synthesizable hardware counterparts and their co-simulation infrastructure.

``src/napl/utils/``
   Shared validation, tensor, device, timing, and test helpers.

Public API
----------

``napl/__init__.py`` re-exports the public classes from the simulation
subpackages. Applications can use top-level imports such as::

   from napl import accuracy, encode, linear, mul_gaines

Deep imports such as ``from napl.sim.operation import mul_gaines`` are also
supported.

Current boundaries
------------------

NAPL does not currently include a Python-to-PyTorch or Python-to-RTL
transpiler. The command-line entry point prints a banner only. The FFT and
biological structure packages contain incomplete interfaces. RTL coverage is
limited to concrete operation folders under ``src/napl/imp/operation/``.
