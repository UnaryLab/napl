Architecture
============

NAPL is a PyTorch framework for programmable spike processing. It supports
per-timestep spike-stream simulation and non-streaming binary-domain kernels.
The Python model defines functional behavior. A subset of spike operations and
streaming layers has verified Verilog-2001 counterparts.

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

Non-streaming kernels process a complete tensor in one call. They use quantized
fixed-point or floating-point arithmetic without an explicit timestep stream.

Hybrid unary-binary (HUB) wrappers, the ``*_hub`` classes, take and return
numeric tensors at their boundary codecs and run one-bit spike streams inside,
one timestep per inner call. One call of their decorated ``forward()`` is a
complete fresh run of ``timestep`` cycles.

Package map
-----------

``src/napl/sim/base/``
   Shared lifecycle, timestep decorators, hardware metadata, and dtype
   configuration.

``src/napl/sim/module/``
   Neural layers and recurrent cells.

``src/napl/sim/operation/``
   Encoders, decoders, arithmetic, comparison, activation, state,
   polarity-conversion, and stream synchronization primitives.

``src/napl/sim/metric/``
   Progressive stream observers and stability stream construction.

``src/napl/sim/algorithm/``
   Compositions of modules and operations, including FFT butterflies and
   streaming radix-2 FFTs.

``src/napl/sim/structure/``
   Placeholder boundaries for biological-neuron abstractions.

``src/napl/imp/``
   Synthesizable hardware counterparts and their co-simulation infrastructure.

``src/napl/utils/``
   Shared validation, tensor, device, timing, and test helpers.

Public API
----------

Each public class or function is imported from the simulation subpackage that
defines it, using the form ``from napl.sim.<subpackage> import <name>``::

   from napl.sim.operation import encode, mul_gaines
   from napl.sim.module import linear_mix
   from napl.sim.metric import accuracy

The subpackages are ``base``, ``operation``, ``module``, ``metric``,
``structure``, and ``algorithm``.

Current boundaries
------------------

NAPL does not currently include a Python-to-PyTorch transpiler. The command-line
entry point prints a banner only, and the biological structure package contains
placeholder interfaces. RTL coverage includes the concrete operation and module
folders under ``src/napl/imp/``; algorithms and metrics have no RTL trees.
