Workflow
========

napl realizes programmable spike processing as a stack of layers: a Python
*simulation* model that defines what every class computes, a Verilog
*implementation* that lowers a subset of those classes to verified hardware, and
a *synthesis* path that lowers a whole user program to RTL. The :doc:`psp_101`
page covers PSP in general; this page covers how napl builds it.

Simulation
----------

The Python model is the definition of what each class computes. It lives under
``src/napl/sim/``, split into subpackages:

``base/``
   Shared module lifecycle, timestep decorators, hardware metadata, and dtype
   configuration. :class:`napl.sim.base.napl_base` is the base class every
   simulation module inherits. See :doc:`../api/sim/base`.

``operation/``
   Encoders, decoders, and the small building blocks: arithmetic, comparison,
   activation approximations, stateful elements, polarity conversion, and
   stream synchronization. ``encode`` and ``decode`` live here. See
   :doc:`../api/sim/operation`.

``module/``
   Tensor-shaped neural layers and recurrent cells: linear, convolution,
   pooling, and gated-recurrent variants, in streaming, FXP, and HUB forms. See
   :doc:`../api/sim/module`.

``metric/``
   Stream observers and stream construction. See :doc:`../api/sim/metric`.

``algorithm/``
   Compositions of modules and operations, including FFT butterflies and
   streaming radix-2 FFTs. See :doc:`../api/sim/algorithm`.

``structure/``
   Reserved boundaries for biological-neuron abstractions (axon, soma, dendrite,
   synapse, receptor, column). The files are empty placeholders with no public
   classes yet. See :doc:`../api/sim/structure`.

Execution models
~~~~~~~~~~~~~~~~~

Every class uses one of two execution models, and the model tells you how to
call it. The choice is exposed by the ``streaming`` class attribute on
:class:`napl.sim.base.napl_base`. Both models operate on PyTorch tensors and run
on supported CPU and GPU devices.

*Streaming* (``streaming = True``) models time-explicit dataflow: spike values
and internal state advance through the network one timestep at a time. You call
the module once per timestep, and each call consumes one bit from each input
stream. A typical loop calls the whole path once per timestep::

   spike_a = encoder_a(value_a)
   spike_b = encoder_b(value_b)
   spike_y = operation(spike_a, spike_b)
   value_y = decode(spike_y)

Run this loop for as many timesteps as your target accuracy needs; the decoder
refines its estimate as more spikes arrive.

*Non-streaming* (``streaming = False``) processes a complete tensor per call and
leaves ``timestep_cur`` at zero. This family covers the fixed-point (FXP) tensor
kernels and the ``*_hub`` wrappers. A *hybrid unary-binary* (HUB) wrapper is
non-streaming at its numeric ports but runs spike streams internally: one call
encodes the numeric input, drives its streaming core for ``timestep`` cycles,
and decodes the result.

The base contract
~~~~~~~~~~~~~~~~~

The shared behavior comes from :class:`napl.sim.base.napl_base`:

- Calling a streaming module runs one timestep and advances its internal counter
  ``timestep_cur`` by one, which is the number of timesteps seen since the last
  reset.
- :attr:`~napl.sim.base.napl_base.valid` is ``True`` once at least one timestep
  has run since construction or the last reset. Reading a result before that
  returns zeros rather than dividing by an empty run.
- :meth:`~napl.sim.base.napl_base.forward_timestep` runs exactly one timestep
  without resetting, so you can step a run and read the progressively refined
  result after each step.
- :meth:`~napl.sim.base.napl_base.reset` returns ``timestep_cur`` and all module
  state to their initial values, resets any child modules, and clears the
  class's own accumulated counts. Because reset returns the run to a known start,
  feeding the same stream again reproduces the same result, which is what makes a
  run repeatable and testable.

Verifying a unit
~~~~~~~~~~~~~~~~

The Python model is the reference for every operation, module, and metric. A
test that checks an approximate result, using a known answer worked out by hand
or an analytic reference, prints the measured error every run for you to inspect
rather than asserting a fidelity bound on it. What a test asserts is the
structural contract a run depends on: a streaming call advances exactly one
timestep, a non-streaming call processes a full tensor without advancing it,
resetting then replaying a stream reproduces the same outputs (bit-exact on
integer streams), and the same result holds across every available device. For
trainable operations that use a straight-through estimator, the test also checks
each gradient against its defining equation.

Every test file is a standalone script with an ``if __name__ == '__main__'``
entry point and is also a valid pytest target, so you verify one unit by running
its focused test through the project environment::

   conda run -n napl python tests/operation/test_mul_gaines.py

Replace ``operation`` with the unit's layer and ``mul_gaines`` with its name.
The focused test is the normal way to verify a change; a full sweep over every
test exists at ``tests/sweep_test.py`` but is meant for a wide regression pass,
not routine checking.

Implementation
--------------

For a subset of spike operations and streaming layers, napl also ships a
verified Verilog-2001 counterpart under ``src/napl/imp/``, with an ``operation/``
RTL tree paralleling ``sim/operation/`` and a ``module/`` RTL tree paralleling
``sim/module/``. Only a subset of operations and modules has RTL, not the whole
simulation model. Each RTL module is verified *bit-exact* against the Python
model by golden-vector co-simulation, so results you get in Python are the
results the hardware produces. How the Verilog matches the Python model, the
repository layout, and how that correspondence is co-simulated are covered in the
:doc:`../implementation/overview`, :doc:`../implementation/tree`, and
:doc:`../implementation/verification` pages.

Synthesis
---------

The synthesis path lowers a whole user program, rather than a single library
class, to RTL. ``src/napl/syn/translate.py`` translates a napl computation graph
(a list of nodes, each a class plus its configuration) into RTL bindings,
resolving each node's Verilog module, parameters, and port map through the
mapping contract in ``src/napl/imp/mapping.yaml``. Full synthesis documentation
is a placeholder; see :doc:`../synthesis/overview`.
