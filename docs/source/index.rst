About
=====

NAPL is UnaryLab's PyTorch framework for programmable spike processing. It
supports per-timestep spike-stream simulation, single-shot binary-domain
kernels, and verified Verilog counterparts for selected operations.

This site separates system design from the generated API reference. Design
pages explain the contracts between components. API pages import NAPL and
render signatures and docstrings from the source tree.

.. toctree::
   :maxdepth: 2
   :caption: Design

   design/architecture
   design/execution
   design/state_and_metrics
   design/hardware

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/verification

.. toctree::
   :maxdepth: 2
   :caption: API

   Reference <api/index>

Canonical project policies remain in `ARCHITECTURE.md`_, `RULE_SIM.md`_, and
`RULE_IMP.md`_ at the repository root.

.. _ARCHITECTURE.md: https://github.com/UnaryLab/napl/blob/main/ARCHITECTURE.md
.. _RULE_SIM.md: https://github.com/UnaryLab/napl/blob/main/RULE_SIM.md
.. _RULE_IMP.md: https://github.com/UnaryLab/napl/blob/main/RULE_IMP.md
