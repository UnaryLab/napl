Overview
========

This page is the layer map of NAPL's simulation stack: what each of the six
:mod:`napl.sim` subpackages is, and how they stack. For the run model itself
(streaming vs. non-streaming execution, the reset/timestep contract, how you
call a class, how a unit is verified) see :doc:`../design/workflow`; for the
per-spike-processing concept see :doc:`../design/psp_101`. The full per-class
detail lives in the :doc:`Reference API <index>`.

The layers
----------

**base** -- the shared contract. :class:`napl.sim.base.napl_base` is the base
class every simulation class inherits. It carries the lifecycle (per-timestep
call, reset) and the hardware metadata that the run model builds on. Reference:
:doc:`sim/base`.

**operation** -- the streaming primitives and the codecs. The arithmetic,
comparison, activation, stateful, polarity, and synchronization building blocks
that process a spike stream one timestep at a time, plus the codecs
:class:`napl.sim.operation.encode` and :class:`napl.sim.operation.decode` that
convert between a number and a spike stream. This layer is self-contained: it
imports nothing from the layers above it. Reference: :doc:`sim/operation`.

**module** -- composed layers built *from* operations. Tensor-shaped neural
layers and recurrent cells (linear, convolution, pooling, and gated-recurrent),
for example :class:`napl.sim.module.linear_mix`. Each module wires operation
primitives into a layer. Reference: :doc:`sim/module`.

**metric** -- measurement instruments that observe a stream (or construct one):
:class:`napl.sim.metric.accuracy`, :class:`napl.sim.metric.correlation`, and the
stability family. Metrics sit to the side of the stack, not on top of it: they
attach to any layer to measure what its stream is doing. Reference:
:doc:`sim/metric`.

**structure** -- currently a placeholder. The files exist (axon, soma, synapse,
dendrite, receptor, column) but are empty; the subpackage exports no public
classes yet. Reserved for biological-neuron abstractions. Reference:
:doc:`sim/structure`.

**algorithm** -- larger compositions built from modules: the FFT family,
butterflies and streaming radix-2 FFTs. Reference: :doc:`sim/algorithm`.

Moving up the stack
-------------------

Each layer composes the one below it: **operation -> module -> algorithm**.
Operations are the streaming primitives; a module wires operations into a
tensor-shaped layer; an algorithm wires modules into a larger computation (an
FFT is butterflies built from multiply and add operations). Every class on this
path inherits :class:`napl.sim.base.napl_base`, so it shares one lifecycle and
one hardware-metadata contract.

**metric** stands apart. A metric does not sit on the operation-module-algorithm
path; it observes any layer on that path, measuring a stream's accuracy,
correlation, or stability wherever you attach it.
