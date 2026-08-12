Overview
========

Synthesis in napl means taking a *program* of napl classes down to hardware:
starting from a computation expressed as napl operations and modules and
lowering it to a synthesizable Verilog realization.

Two paths cover this at different scopes, at different levels of maturity.

napl-to-RTL
    The real, existing lowering. It binds each napl class in a computation
    graph to its verified Verilog counterpart under ``src/napl/imp/``, through
    the per-class mapping contract in ``src/napl/imp/mapping.yaml``. The Python
    model in ``src/napl/syn`` walks the graph and, for each node, fixes the RTL
    module, its Verilog parameters, and its port map. See :doc:`napl_to_rtl`.

user-to-system
    The future, whole-program path: lowering a whole user program to a full
    system's RTL, beyond single library classes. See :doc:`user_to_system`.

Current maturity, stated honestly: the napl-to-RTL translation exists in code
and is exercised by tests. Full user-facing synthesis documentation and the
whole-system path are planned; this section grows as they land.
