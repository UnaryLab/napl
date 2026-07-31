Execution Models
================

NAPL classes use one of two execution models. Both operate on PyTorch tensors
and can use supported CPU and GPU devices.

Streaming execution
-------------------

Streaming classes inherit ``napl.sim.base.napl_base`` and keep the default
``streaming = True``.

* One call represents one timestep and one bit from each input stream.
* ``napl_base.__call__`` increments ``timestep_cur`` before ``forward()``.
* Stateful classes update persistent state across calls.
* ``valid`` becomes true after the first timestep following construction or
  reset.
* ``reset()`` restores the timestep and class state to their initial values.

The ``napl.sim.base.napl_sim_timesteps`` decorator repeats a module method.
``napl.sim.base.napl_sim_timesteps_func`` provides the same behavior for a
free function. Both accept ``timesteps`` as a keyword argument. The decorated
body still implements one timestep.

A typical streaming loop is::

   spike_a = encoder_a(value_a)
   spike_b = encoder_b(value_b)
   spike_y = operation(spike_a, spike_b)
   value_y = decoder(spike_y)

The complete path runs once per timestep.

Single-shot execution
---------------------

Single-shot classes set ``streaming = False``.

* One call processes the complete tensor.
* ``napl_base.__call__`` does not increment ``timestep_cur``.
* ``timestep_cur`` remains zero and ``valid`` remains false.
* Trainable HUB, FXP, TLUT, and hard neural kernels can use custom
  ``torch.autograd.Function`` implementations and straight-through
  estimators.

Single-shot execution models binary-domain computation. It does not imply that
a matching RTL module exists.

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
