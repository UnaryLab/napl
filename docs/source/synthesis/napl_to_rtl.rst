napl-to-RTL
===========

The napl-to-RTL path is the existing lowering of verified library classes to
their Verilog counterparts. It lives in ``src/napl/syn/translate.py`` and is
driven by the mapping contract in ``src/napl/imp/mapping.yaml``.

What ``translate.py`` produces
------------------------------

The translator takes a napl *computation graph*: a sequence of nodes, each a
mapping that names a napl ``class`` (a class object or its dotted/path string)
plus a ``config`` dictionary carrying that node's configuration. It resolves
each node against the mapping table and returns an ``RtlBinding`` per node.
``translate_node`` handles a single node, ``translate_graph`` a sequence of
nodes in input order, and ``translate`` dispatches to whichever the argument
looks like.

An ``RtlBinding`` is a *binding structure*, not Verilog text. The translator
does not emit or generate any RTL source; it resolves, for each node:

* ``rtl_module`` - the name of the target Verilog module,
* ``file_path`` - the resolved path to that module's ``.v`` file under
  ``src/napl/imp/``,
* ``parameters`` - the Verilog parameter values, evaluated from the node's
  config,
* ``port_map`` - a ``PortMap`` grouping the module's ``inputs`` and ``outputs``
  port names.

Resolving a node walks three steps: select the mapping entry, resolve its
parameters, and build the port map.

Selecting the mapping entry
---------------------------

For a node's class name, the translator collects the mapping entries whose
``sim_module`` basename matches (falling back to a match on ``rtl_module``).
When a class has several RTL variants, selection narrows by an explicit
``rtl_module`` on the node or in its config, then by a ``polarity`` (matching
an entry named ``<class>_<polarity>``), then by an exact ``rtl_module ==
<class>`` entry, then by there being only one entry. If a class has multiple
variants and none of these disambiguate, translation raises
``TranslationError`` naming the available variants.

The mapping contract (``mapping.yaml``)
---------------------------------------

``mapping.yaml`` is a list of entries, one per RTL variant, that serves as the
per-class contract between a napl class and its hardware. Each entry uses these
fields (the ones present in the file):

* ``rtl_module`` - the Verilog module name and ``.v`` file stem.
* ``layer`` - the ``imp/`` subtree holding the RTL, one of ``operation`` or
  ``module`` (defaulting to ``operation``); it selects between
  ``src/napl/imp/operation/`` (counterpart of ``sim/operation/``) and
  ``src/napl/imp/module/`` (counterpart of ``sim/module/``).
* ``sim_module`` - the source napl simulation file this RTL realizes, e.g.
  ``sim/operation/mul_gaines.py`` or ``sim/module/linear_ugemm.py``.
* ``inputs`` / ``outputs`` - maps from the logical port name to the Verilog
  port identifier (for example ``input: i_input``, ``output: o_output``). A
  port mapped to ``null`` marks a logical input the variant has no hardware
  port for.
* ``parameters`` - a map from Verilog parameter name to an expression string
  evaluated per node (for example ``WIDTH: "config['width']"`` or ``ENTRY:
  input.size(dim)``). Empty when the module takes no parameters.
* ``requires`` - optional guard expressions (a string or a list) that must all
  evaluate truthy for the configuration to be supported; a false guard raises
  ``TranslationError``. Guards encode what a variant does not implement, such
  as ``config['fracwidth'] == 0`` or ``WIDTH <= 30``.

The file resolves classes across both layers: for example, the operation-layer
``mul_gaines`` role maps to ``mul_gaines_bipolar`` / ``mul_gaines_unipolar``,
and the module-layer ``linear_ugemm`` role maps to ``linear_ugemm_bipolar`` /
``linear_ugemm_unipolar``. The same contract is described from the RTL side in
:doc:`../implementation/tree`.

Parameter expressions and the restricted evaluator
---------------------------------------------------

Parameter and ``requires`` expressions are not evaluated with Python ``eval``.
``translate.py`` implements ``_RestrictedEvaluator``, which parses each
expression with ``ast`` and walks only an allowed subset: literals, names bound
in the evaluation context, list/tuple, subscripting, the arithmetic and
comparison and boolean operators, conditional (``if``/``else``) expressions,
and a fixed set of functions - ``ceil``, ``floor``, ``log2``, ``len``,
``int``, ``get``, ``shape``, ``area`` - plus the single attribute call
``input.size(dim)``. Any other name, call, or syntax raises
``TranslationError``.

The evaluation context is built per node from its config. It binds the node's
config keys, the whole ``config`` mapping under the name ``config``, an
``input`` shape object exposing ``.size(dim)``, the reduction ``dim``, and a
``SEGMENT`` count. Parameters are evaluated in order and each resolved value is
added back to the context, so a later parameter (for example an accumulator
width) can refer to an earlier one (for example ``ENTRY``). After the
parameters resolve, the ``requires`` guards are checked in the same context.

What this path does and does not do
-----------------------------------

It does resolve each node to a concrete RTL module file, a set of evaluated
Verilog parameter values, and a port map, and it rejects configurations a
variant cannot implement. It does not emit Verilog, elaborate a netlist, or
connect nodes into a top-level module; the per-node bindings are the output.
The Verilog it binds to is the verified RTL under ``src/napl/imp/``, whose
verification against the napl model is described in
:doc:`../implementation/verification`.
