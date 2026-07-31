State and Metrics
=================

The canonical simulation policy is `RULE_SIM.md
<https://github.com/UnaryLab/napl/blob/main/RULE_SIM.md>`_. The requirements
below summarize its state and metric contracts.

Lifecycle and reset
-------------------

Streaming state belongs to the class that updates it.
``napl.sim.base.napl_base.reset()`` resets ``timestep_cur`` and registered
child modules before calling the subclass ``_reset()`` hook. Every
``napl_base`` subclass under ``src/napl/sim/`` defines ``_reset()`` explicitly.
A class without local mutable state uses an empty hook::

   def _reset(self):
       pass

A stateful hook restores only local state to the values used at construction.
Reset followed by replay of the same inputs must reproduce the same outputs.

Persistent tensors
------------------

``torch.nn.Parameter`` is reserved for trainable weights and biases.
Non-trainable persistent tensors are registered with ``register_buffer`` so
that device migration and ``state_dict`` handling remain consistent. Buffer
updates do not use ``.data``.

State updates follow these forms:

* ``zero_()`` for zero resets and ``fill_()`` for constant resets.
* In-place arithmetic such as ``add_()``, ``sub_()``, and ``clamp_()`` in
  steady-state hot paths.
* ``copy_()`` for snapshots, alias prevention, delayed state, returned data,
  or required object identity.
* ``resize_()`` or ``resize_as_()`` followed by an in-place update when state
  shape changes.

Direct reassignment to a buffer name is limited to measured hot-path benefits
or simpler shape-changing updates. The assigned tensor is detached, and the
implementation must preserve buffer registration, device migration, reset and
replay, and ``state_dict`` serialization.

Metric analysis
---------------

Metrics are streaming observers. Each call consumes the current spike state.
An ``analyze()`` method returns ``(value, result)``:

``value``
   The metric's primary per-element tensor.

``result``
   The complete ``napl.sim.metric._shared.Analysis`` value returned by the
   shared analysis helper.

Derived analysis outputs are local values. Metrics do not store them on
``self``, register them as buffers, or include them in ``state_dict``. Callers
select required summary fields from the returned value, for example::

   value, result = metric.analyze()
   index = result.max_absolute_index

``napl.sim.metric.accuracy``, ``napl.sim.metric.correlation``,
``napl.sim.metric.stability``, ``napl.sim.metric.stability_flux``, and
``napl.sim.metric.stability_norm`` follow this interface.
