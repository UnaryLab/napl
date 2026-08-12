About
=====

NAPL is UnaryLab's PyTorch framework for programmable spike processing. It
supports per-timestep spike-stream simulation, non-streaming binary-domain
kernels, and verified Verilog counterparts for selected operations.

This site separates system design from the generated API reference. Design
pages explain the contracts between components. API pages import NAPL and
render signatures and docstrings from the source tree.

.. toctree::
   :maxdepth: 2
   :caption: Design

   design/overview
   design/psp_101
   design/workflow

.. toctree::
   :maxdepth: 2
   :caption: Simulation

   Overview <api/overview>
   Reference API <api/index>

.. toctree::
   :maxdepth: 2
   :caption: Implementation

   implementation/overview
   implementation/tree
   implementation/verification

.. toctree::
   :maxdepth: 2
   :caption: Synthesis

   synthesis/overview
   synthesis/napl_to_rtl
   synthesis/user_to_system

For how NAPL is verified, see :doc:`Workflow <design/workflow>` for the Python
model and how to check a unit, and :doc:`Verification <implementation/verification>`
for the bit-exact hardware co-simulation.
