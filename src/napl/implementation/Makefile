# NAPL RTL -- build & test the Verilog implementation of an operation.
#
#   make test  OP=mul_and     # generate vectors, compile, simulate, check
#   make clean OP=mul_and     # remove that op's build artifacts + generated vectors
#
# Each operation is self-contained under its own folder:
#   <op>/rtl/    hand-written RTL modules (one per variant)
#   <op>/tb/     <op>_tb.v self-checking testbench
#   <op>/gen/    gen_<op>.py -- golden vectors from the napl Python model
#   <op>/vec/    generated <op>.vec (gitignored)
#   <op>/build/  compiled sim + log (auto-created, gitignored)
#
# Vector generation imports napl, so run inside the `napl` conda env
# (or:  conda run -n napl make test OP=mul_and).
# Requires the Icarus Verilog toolchain (iverilog / vvp) on PATH.

OP       ?= mul_and
PYTHON   ?= python
IVERILOG ?= iverilog
VVP      ?= vvp

.PHONY: test vectors compile run clean

test: run

# Golden vectors come from the Python functional model.
vectors:
	$(PYTHON) $(OP)/gen/gen_$(OP).py

compile: vectors
	mkdir -p $(OP)/build
	$(IVERILOG) -g2012 -Wall -o $(OP)/build/sim $(OP)/rtl/*.v $(OP)/tb/$(OP)_tb.v

# vvp runs from inside the op folder so the testbench's relative path
# (vec/<op>.vec) resolves.
run: compile
	cd $(OP) && $(VVP) build/sim | tee build/sim.log
	@grep -q "^PASS" $(OP)/build/sim.log || { echo "*** $(OP): TEST FAILED ***"; exit 1; }

clean:
	rm -rf $(OP)/build $(OP)/vec/*.vec
