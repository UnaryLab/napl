State and Metrics
=================

The canonical simulation policy is `RULE_SIM.md
<https://github.com/UnaryLab/napl/blob/main/RULE_SIM.md>`_. The requirements
below summarize its state and metric contracts.

Lifecycle and reset
-------------------

Streaming state belongs to the class that updates it, while
:class:`napl.napl_base` provides the common lifecycle. The public reset
contract is documented by :meth:`napl.napl_base.reset`; class-specific state
behavior belongs to each class's API documentation. The implementation and
verification requirements remain in the canonical simulation policy linked
above.

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

Metrics separate streaming observation from explicit analysis. The metric API
pages own their call, state, and analysis contracts: :class:`napl.accuracy`,
:class:`napl.correlation`, :class:`napl.stability`,
:class:`napl.stability_flux`, and :class:`napl.stability_norm`. Shared summary
calculation is listed as :func:`napl.analyze`.
