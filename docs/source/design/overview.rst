Overview
========

Programmable spike processing (PSP) computes with numbers carried as streams of
spikes and composes small operations into a processing pipeline. This section
introduces PSP and how NAPL builds it, in two pages:

- :doc:`psp_101` is the concept: what a spike stream is, the
  encode-process-decode workflow, what you can measure about a stream, and the
  kinds of problem PSP suits. Start here if you are deciding whether PSP fits
  your work.
- :doc:`workflow` is the realization: the layers of NAPL's stack, the streaming
  and non-streaming execution models, the shared base contract, and how a unit
  is verified. Read it once you want to build with NAPL.

The rest of the documentation continues the arc. The :doc:`Simulation overview
<../api/overview>` maps the six ``napl.sim`` layers to their reference pages, the
:doc:`Implementation overview <../implementation/overview>` covers the verified
Verilog counterpart, and the :doc:`Synthesis overview <../synthesis/overview>`
covers lowering a program to hardware.
