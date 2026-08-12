Overview
========

The Python simulation model is the source of truth for what each class computes.
For a subset of classes, NAPL also ships a synthesizable Verilog-2001 circuit
that produces bit-for-bit the same outputs. This page explains how the two line
up and why you can trust that the hardware matches the Python model.

The Python model is the reference
---------------------------------

NAPL's Python simulation defines the intended behavior of every operation,
module, and metric. Where a Verilog hardware counterpart exists, it is held to
that model: the RTL is not an independent design that happens to agree, it is
required to reproduce the Python class output on every golden vector. The bar is
bit-exact, so the hardware and the reference model are the same function, not
merely close. The :doc:`verification` page describes the check that enforces
this.

The ``src/napl/imp/`` tree holds Verilog for the concrete operations and
streaming layers under ``src/napl/imp/operation/`` and
``src/napl/imp/module/``. The metric and algorithm layers have no hardware
counterparts. The :doc:`tree` page walks that directory.

Signals and cycles
-------------------

An operation's RTL is a scalar circuit; a streaming-layer RTL replicates it
across tensor lanes. Each spike port carries the corresponding Python
``forward()`` argument name with an ``i_`` (input) or ``o_`` (output) prefix.
One Python ``forward()`` timestep corresponds to one rising clock edge. The
active-low reset ``i_rst_n`` restores every register to the exact state the
Python model has right after :meth:`~napl.sim.base.napl_base.reset`, which is
not necessarily all-zero.

Pipeline delay
--------------

Some circuits are registered and take a few cycles from input to output. That
*pipeline delay*, the number of clock cycles before an input affects the output,
is recorded on the Python class as ``self.hw.pp_delay`` (0 for a purely
combinational operation). The value matches the verified RTL pipeline, so it also
tells you how to align streams when two paths reconverge. The
:class:`napl.sim.base.napl_base` ``hw`` field also carries per-corner timing
numbers in nanoseconds, which describe an implementation but do not change
functional results.

Next reads
----------

- :doc:`tree` maps the ``src/napl/imp/`` directory and the ``mapping.yaml``
  parameter contract.
- :doc:`verification` shows how to run the bit-exact check yourself.
