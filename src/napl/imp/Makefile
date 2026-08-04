# Run `conda run -n napl make test OP=<op>` with iverilog and vvp on PATH.
# Each operation supplies RTL, a self-checking testbench, and Python golden vectors.

OP       ?= mul_gaines
PYTHON   ?= python
IVERILOG ?= iverilog
VVP      ?= vvp

.PHONY: test vectors compile run clean

test: run

vectors:
	$(PYTHON) operation/$(OP)/gen/gen_$(OP).py

compile: vectors
	mkdir -p operation/$(OP)/build
	$(IVERILOG) -g2001 -Wall -Ioperation -o operation/$(OP)/build/sim operation/$(OP)/rtl/*.v operation/$(OP)/tb/$(OP)_tb.v

# Testbenches resolve vec/<op>.vec relative to the operation directory.
run: compile
	cd operation/$(OP) && $(VVP) build/sim | tee build/sim.log
	@grep -q "^PASS" operation/$(OP)/build/sim.log || { echo "*** $(OP): TEST FAILED ***"; exit 1; }

clean:
	rm -rf operation/$(OP)/build operation/$(OP)/vec/*.vec
