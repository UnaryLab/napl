Hardware Boundary
=================

The Python simulation model is the source of functional behavior. The
``src/napl/imp/`` tree contains Verilog-2001 counterparts for a subset of
concrete operations and streaming layers. Metrics and algorithms do not
currently have matching RTL trees.

The root `RULE_IMP.md
<https://github.com/UnaryLab/napl/blob/main/RULE_IMP.md>`_ is the canonical
hardware design and verification policy.

RTL mapping
-----------

Each implemented operation or module has a self-contained directory under
``src/napl/imp/operation/<op>/`` or ``src/napl/imp/module/<module>/``::

   <op>/
   |-- rtl/    synthesizable modules
   |-- tb/     self-checking testbench
   |-- gen/    Python golden-vector generator
   |-- vec/    generated vectors and parameter header
   `-- build/  generated simulation output

Operation folder and RTL module names match their simulation counterparts.
Module folders retain their established hardware base names and are selected by
``src/napl/imp/mapping.yaml``. Polarity-dependent circuits use separate
``_unipolar`` and ``_bipolar`` modules. Other circuit choices use Verilog
parameters when they are fixed at elaboration.

Signals and cycles
------------------

An RTL operation is a scalar circuit. Streaming-layer RTL replicates those
circuits across tensor lanes. Spike ports use the corresponding Python
``forward()`` argument name with an ``i_`` or ``o_`` prefix. One Python
``forward()`` timestep corresponds to one positive clock edge.

Active-low ``i_rst_n`` restores every register to the exact Python post-reset
state. This state is not necessarily zero.

Timing metadata
---------------

Timing metadata connects RTL-backed Python classes to verified latency and
per-corner timing results. ``hw_params`` and ``timing`` own
the public field definitions; the hardware policy linked above owns the
cross-model verification requirements.

Golden-vector contract
----------------------

Expected RTL outputs come from the matching NAPL Python class. The generator emits
the vectors and a Verilog parameter header from one Python configuration. The
testbench includes that header and applies the same sizing values to the RTL.

A self-checking testbench compares every output on its corresponding cycle and
exits with failure on any mismatch. Stateful tests cover the first post-reset
cycle, reset after state changes, and replay of the same inputs.
