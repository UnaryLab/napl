Execution Models
================

NAPL classes use one of two execution models. Both operate on PyTorch tensors
and can use supported CPU and GPU devices.

Streaming execution
-------------------

Streaming classes model time-explicit dataflow: state and spike values advance
through a composed network one timestep at a time. The shared lifecycle is
defined by :class:`napl.napl_base`; see its :meth:`~napl.napl_base.__call__`,
:attr:`~napl.napl_base.valid`, and :meth:`~napl.napl_base.reset` API entries.
The :func:`napl.napl_sim_timesteps` decorator repeats module methods and free
functions.

A typical streaming loop is::

   spike_a = encoder_a(value_a)
   spike_b = encoder_b(value_b)
   spike_y = operation(spike_a, spike_b)
   value_y = decoder(spike_y)

The complete path runs once per timestep.

Single-shot execution
---------------------

Single-shot classes expose binary-domain tensor computation through the same
module composition model. This execution family is independent of RTL
availability. Constructor and call behavior belong to each class's API page.

Kernel families
---------------

Streaming kernels
   Unsuffixed neural modules and spike operations process one-bit streams one
   timestep at a time.

HUB kernels
   ``*_hub`` classes use hybrid unary-binary arithmetic on complete tensors.

FXP kernels
   ``*_fxp`` classes and ``round_fxp`` use quantized fixed-point tensors.

TLUT kernels
   ``*_tlut`` classes use table-based quantized computation.

Hard neural kernels
   ``mgu_hard*`` and ``gru_hard*`` classes use binary tensors and hard
   nonlinearities.

Encoding contracts
------------------

Unipolar streams represent values in ``[0, 1]``. Bipolar streams represent
values in ``[-1, 1]`` and map a value ``x`` to probability ``(x + 1) / 2``.

Operands that require statistical independence must use independent number
sequences. Sobol streams use distinct ``dim`` values for independent operands.
Bipolar zero-padding in ``napl.sim.module.conv`` uses a separate decorrelated
rate-0.5 stream.
