Verification
============

Where a Verilog counterpart exists, it is verified by *golden-vector
co-simulation*, and the bar is bit-exact. The Python model generates the
expected input and output vectors from the class's fidelity inputs, then the
Verilog RTL and a self-checking testbench are compiled and simulated with Icarus
Verilog, and the testbench replays the vectors and asserts the hardware output
against the model's vectors cycle for cycle. Because the expected values come
from the Python class itself, a passing co-simulation means the Verilog
reproduces the Python model cycle for cycle.

The flow for one unit is:

1. The Python golden-vector generator in the unit's ``gen/`` directory runs the
   NAPL class on a chosen configuration and emits both the expected output
   vectors and a Verilog header of the sizing parameters.
2. The testbench includes that header, applies the same sizing to the RTL, and
   drives the RTL with the same inputs.
3. The self-checking testbench compares every RTL output against the Python
   expected value on its corresponding cycle and fails on any mismatch.

Stateful circuits are additionally checked on the first post-reset cycle, after
a reset that follows state changes, and on a replay of the same inputs.

Running the check
-----------------

Icarus Verilog's ``iverilog`` and ``vvp`` must be on your ``PATH``. Run from the
``src/napl/imp/`` directory. For an operation-layer unit
(:class:`napl.sim.operation.mul_gaines`)::

   conda run -n napl make test OP=mul_gaines

For a module-layer unit (:class:`napl.sim.module.linear_mix`)::

   conda run -n napl make test MODULE=linear_mix

Each generates the vectors from the Python model, builds the RTL and testbench,
runs the simulation, and prints a ``PASS`` line reporting a full vector match; a
mismatch, a simulation error, or a missing ``PASS`` line turns the target red.

Supporting checks
-----------------

These targets are unit-independent and take no ``OP=`` or ``MODULE=``:

- ``conda run -n napl make guards`` elaborates every guarded module with a
  violating parameter set and requires the build to fail, so a dropped
  parameter guard turns red (runs ``test_guards.py``).
- ``conda run -n napl make equiv`` walks each inlined circuit copy against its
  standalone reference module (runs ``test_equiv.py``).
- ``conda run -n napl make floor`` requires the globbed unit list to match the
  ``mapping.yaml`` registry, so a deleted or renamed unit folder turns red
  (runs ``sweep_floor.py``). See :doc:`tree` for that registry.

A full ``make sweep`` target also exists, running every unit plus ``floor``,
``guards``, ``equiv``, and the translate pytest. It is the whole-suite
regression pass, run only on explicit request, not a routine step.
