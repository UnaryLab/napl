Execution Models
================

NAPL classes use one of two execution models. Both operate on PyTorch tensors
and can use supported CPU and GPU devices.

Streaming execution
-------------------

Streaming classes model time-explicit dataflow: state and spike values advance
through a composed network one timestep at a time. The shared lifecycle is
defined by :class:`napl.sim.base.napl_base`; see its
``__call__``, :attr:`~napl.sim.base.napl_base.valid`,
and :meth:`~napl.sim.base.napl_base.reset` API entries.
The ``napl_sim_timesteps`` decorator repeats module methods and free
functions, whose bodies represent one timestep. On a :class:`napl.sim.base.napl_base`
module carrying its own run length as ``timestep``, such as a ``*_hub`` wrapper,
a call with no ``timesteps`` keyword is a complete fresh run: it resets the
module and its registered children, repeats the body for ``timestep`` cycles on
the same arguments, and returns the final cycle's value. Every other target
passes ``timesteps``, and the decorator then repeats the body that many times
without resetting.
:meth:`~napl.sim.base.napl_base.forward_timestep` drives a decorated ``forward()``
one timestep at a time, incrementing ``timestep_cur`` once per call for a
streaming module and never resetting. A run counts its cycles on
``timestep_cur`` for a streaming module; a non-streaming module such as a
``*_hub`` wrapper keeps its own counter at zero, and the streaming children it
drives hold the count.

A typical streaming loop is::

   spike_a = encoder_a(value_a)
   spike_b = encoder_b(value_b)
   spike_y = operation(spike_a, spike_b)
   value_y = decode(spike_y)

The complete path runs once per timestep.

Non-streaming execution
---------------------

Non-streaming classes set ``streaming = False``, process a complete tensor per
call, and keep ``timestep_cur`` at zero. They cover binary-domain tensor
computation and the ``*_hub`` wrappers, which are non-streaming at their numeric
ports and run spike streams inside, through the same module composition model.
This execution family is independent of RTL availability. Constructor and call
behavior belong to each class's API page.

Kernel families
---------------

These families cover the computational classes. The ``napl.sim.metric`` classes
``accuracy``, ``correlation``, ``stability``, ``stability_builder``,
``stability_flux``, and ``stability_norm`` are deliberately outside them: they
serve stream analysis, measuring the properties of a stream or building a stream
that has prescribed ones, instead of computing a result from operand streams.

A ``_dyn`` name in the scaling and FFT families marks the variant that takes its
scale at call time rather than at construction. Such a class replaces the fixed
key ``scale`` with the ceiling key ``scale_max``, in its own config or in a
nested one, and each call then supplies the scale in force for that timestep.
Exactly six classes follow that rule: ``add_scale_dyn``, ``div_scale_dyn``,
``butterfly_ugemm_dyn``, ``butterfly_mix_dyn``, ``fft_dyn``, and
``fft_dyn_hub``. Every other class whose name ends in ``_dyn``, such as
``mul_ugemm_dyn``, is outside the rule and takes no scale key.

``add_scale_dyn`` and ``div_scale_dyn`` hold ``scale_max`` in their own config,
so ``napl_base`` both requires it and rejects ``scale`` as a key outside the
accepted list. The other four take it in the nested ``add_config``, while their
own ``mul_config`` or ``codec_config`` accepts the same keys as the static
counterpart's. An ``add_config`` carrying ``scale`` alone fails their own guard
with ``Missing key <scale_max> in the dynamic adder configuration``; the
``napl_base`` rejection of ``scale`` fires only when both keys are present.

Streaming kernels
   ``avgpool2d_ugemm``, ``conv_gaines``, ``conv_mix``, ``conv_ugemm``,
   ``linear_gaines``, ``linear_mix``, ``linear_ugemm``, ``mgu_hard_mix``, and
   streaming operations process one-bit streams one timestep at a time.

HUB kernels
   ``conv_ugemm_hub``, ``fft_dyn_hub``, ``fft_hub``, ``linear_ugemm_hub``, and
   ``mgu_hard_mix_hub`` take and return numeric tensors at their boundary codecs
   and run one-bit spike streams inside. Each is non-streaming at those ports: a
   call with no ``timesteps`` keyword is a complete fresh run of ``timestep``
   cycles, one timestep per inner call on the same numeric input.

FXP kernels
   ``conv_fxp``, ``linear_fxp``, ``mgu_hard_fxp``, ``relu_fxp``, ``round_fxp``,
   ``sigmoid_fxp``, and ``tanh_fxp`` process one complete quantized fixed-point
   tensor per call.

Binary FFT butterfly
   ``butterfly_fp`` processes one complete complex tensor per call.

Streaming FFT butterfly
   ``butterfly_mix``, ``butterfly_mix_dyn``, ``butterfly_ugemm``, and
   ``butterfly_ugemm_dyn`` process spike streams one timestep at a time.

Streaming FFT
   ``fft`` and ``fft_dyn`` compose butterfly stages into a full radix-2
   transform and process spike streams one timestep at a time. The optional
   ``mul_config`` key ``kernel`` picks the stage class: ``'ugemm'``, the
   default, builds ``butterfly_ugemm`` stages, and ``'mix'`` builds
   ``butterfly_mix`` stages, which hold their own twiddle encoder. The two
   stage classes differ in accuracy and in the keys they accept, so they are
   not interchangeable. ``fft_hub`` and ``fft_dyn_hub`` pass ``mul_config``
   through unchanged and carry the same knob.

Encoding contracts
------------------

Unipolar streams represent values in ``[0, 1]``. Bipolar streams represent
values in ``[-1, 1]`` and map a value ``x`` to probability ``(x + 1) / 2``.

Operands that require statistical independence must use independent number
sequences. Sobol streams use distinct ``dim`` values for independent operands.
Bipolar zero-padding in ``napl.sim.module.conv_mix`` uses a separate decorrelated
rate-0.5 stream.
