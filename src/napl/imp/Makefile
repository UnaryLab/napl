# Run `conda run -n napl make test OP=<op>` (operation layer) or
# `conda run -n napl make test MODULE=<module>` (module layer) with iverilog and
# vvp on PATH. Each unit supplies RTL, a self-checking testbench, and Python
# golden vectors.

# OP defaults to the canonical example; giving MODULE= instead requires OP untouched.
DEFAULT_OP := mul_gaines
OP       ?= $(DEFAULT_OP)
MODULE   ?=
PYTHON   ?= python
IVERILOG ?= iverilog
VVP      ?= vvp

ifneq ($(MODULE),)
ifneq ($(OP),$(DEFAULT_OP))
$(error set OP=<op> or MODULE=<module>, not both)
endif
endif

ifeq ($(MODULE),)
LAYER := operation
UNIT  := $(OP)
# An operation compiles from its own rtl/ only.
LIBS  :=
else
LAYER := module
UNIT  := $(MODULE)
# A module composes operation-layer circuits; iverilog pulls each one in from the
# operation rtl/ directories, where the file name equals the module name.
LIBS  := $(patsubst %,-y %,$(wildcard operation/*/rtl))
endif

.PHONY: test vectors compile run guards clean

# Every unit test runs the elaboration guards first, so a deleted or reversed
# guard turns a target red without anyone typing `make guards`.
test: guards run

vectors:
	$(PYTHON) $(LAYER)/$(UNIT)/gen/gen_$(UNIT).py

compile: vectors
	mkdir -p $(LAYER)/$(UNIT)/build
	$(IVERILOG) -g2001 -Wall -I$(LAYER) $(LIBS) -o $(LAYER)/$(UNIT)/build/sim $(LAYER)/$(UNIT)/rtl/*.v $(LAYER)/$(UNIT)/tb/$(UNIT)_tb.v

# Testbenches resolve vec/<unit>.vec relative to the unit directory.
run: compile
	cd $(LAYER)/$(UNIT) && $(VVP) build/sim | tee build/sim.log
	@grep -q "^PASS" $(LAYER)/$(UNIT)/build/sim.log || { echo "*** $(UNIT): TEST FAILED ***"; exit 1; }
	@# vvp exits 0 after a runtime ERROR line, so an unreadable $$readmemb ROM
	@# would otherwise reach the PASS grep with an all-x memory.
	@! grep -q "^ERROR" $(LAYER)/$(UNIT)/build/sim.log || { echo "*** $(UNIT): SIMULATION ERROR ***"; exit 1; }

# Elaborate every guarded module with a violating parameter set and require the
# build to fail; unit-independent, so it takes no OP= or MODULE=.
guards:
	$(PYTHON) test_guards.py

clean:
	rm -rf $(LAYER)/$(UNIT)/build $(LAYER)/$(UNIT)/vec/*.vec $(LAYER)/$(UNIT)/vec/*_params.vh
