Layout
======

The Verilog counterpart of the Python simulation lives under ``src/napl/imp/``,
split into an ``operation/`` tree for the scalar operation circuits and a
``module/`` tree for the streaming layers built from them. The metric and
algorithm layers have no hardware counterparts.

Unit directories
----------------

Each implemented operation or module is a self-contained unit directory. For
example, ``src/napl/imp/operation/mul_gaines/`` and
``src/napl/imp/module/linear_mix/`` each hold::

   <unit>/
   |-- gen/     Python golden-vector and RTL-parameter generator
   |-- rtl/     synthesizable Verilog module(s)
   |-- tb/      self-checking testbench
   |-- vec/     generated golden vectors and parameter header
   `-- build/   generated simulation output

The RTL module name equals the file name and, for an operation, the Python
operation name (:class:`napl.sim.operation.mul_gaines`). When a circuit's
behavior depends on polarity, there are separate ``_unipolar`` and ``_bipolar``
modules in ``rtl/``; other fixed choices are Verilog parameters set at
elaboration. A streaming-layer unit
(:class:`napl.sim.module.linear_mix`) is a lane replication of the operation
circuits it composes.

The parameter contract
----------------------

``src/napl/imp/mapping.yaml`` is the simulation-class-to-RTL-module registry,
read by ``napl/syn/translate.py``. Each entry fixes, for one unit, its
``rtl_module`` name, its ``layer`` (``operation`` or ``module``), the
``sim_module`` Python source it binds to, the input and output port names, and
the Verilog ``parameters`` as expressions over the simulation ``config`` (for
example ``DEPTH: "config['depth']"``). An entry may also list ``requires``
constraints that a configuration must satisfy to have a hardware form. This file
is what maps a NAPL computation graph onto the circuits and what fixes each
unit's Verilog parameters; the co-simulation coverage floor
(:doc:`verification`) checks the swept unit list against this registry.
