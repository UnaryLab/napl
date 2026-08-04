Verification
============

Simulation changes follow `RULE_SIM.md
<https://github.com/UnaryLab/napl/blob/main/RULE_SIM.md>`_. RTL-backed changes
also follow `RULE_IMP.md
<https://github.com/UnaryLab/napl/blob/main/RULE_IMP.md>`_. These root files are
the canonical policies.

Simulation gates
----------------

Execution model
   Streaming calls advance exactly one timestep. Reset and replay reproduce
   outputs. Single-shot calls process a full tensor without advancing
   ``timestep_cur``.

Device coverage
   The full test body runs on every device returned by
   ``napl.utils._shared_test.devices()``. Modules and inputs move together.

Numerical fidelity
   Tests include known-answer cases and analytic references. Integer spikes
   use exact equality when required. Approximate results state and assert their
   numerical bound.

Gradients
   Trainable operations with straight-through estimators compare each declared
   input and parameter gradient with its defining equation.

Performance
   Tests use identical pre-generated CPU inputs on every device. CPU is the
   baseline. Shared ``benchmark`` and ``timer`` helpers provide repeated and
   synchronized elapsed measurements.

State and public API
   Tests cover boundary values and timesteps, shapes, dtypes, device migration,
   reset and replay, serialization, and public metric state where applicable.

Test entry points
-----------------

Streaming kernels start from ``tests/template_streaming_kernel.py``.
Single-shot trainable kernels start from
``tests/template_single_shot_trainable.py``. Both use suite drivers in
``napl.utils._shared_test``.

Run a focused pytest target through the project environment::

   conda run -n napl pytest tests/operation/test_mul_gaines.py

The default check is the focused test for the changed unit. The full standalone
test sweep is::

   conda run -n napl python tests/sweep_test.py

Assertions are required for correctness, state, reset, and gradient checks.
Printed values alone do not establish a pass. A skipped case names the
unsupported device or feature.

RTL co-simulation
-----------------

An RTL-backed change first passes its Python test. Then run from the repository
root::

   conda run -n napl make -C src/napl/imp test OP=<op>

The command generates expected vectors from the Python model, compiles the RTL
and self-checking testbench with Icarus Verilog, and runs the simulation. A
passing run reports a full vector match and exits zero.

Verification evidence for RTL records the command, configuration and sizing
parameters, vector count, reset cases, observed latency, expected
``self.hw.pp_delay``, and exit status. Generated ``vec/*.vec``,
``vec/*_params.vh``, and ``build/`` content remains untracked.
