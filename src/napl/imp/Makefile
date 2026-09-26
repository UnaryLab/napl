# Run `conda run -n napl make test OP=<op>` (operation layer) or
# `conda run -n napl make test MODULE=<module>` (module layer) with iverilog and
# vvp on PATH. Each unit supplies RTL, a self-checking testbench, and Python
# golden vectors. `make sweep` runs every unit plus the mapping.yaml floor, the guards
# suite, the reference-form equivalence walk, and the translate pytest; it runs
# only on explicit request. The per-change check is `make test` on each changed unit.

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
else
LAYER := module
UNIT  := $(MODULE)
endif

# An operation composes other operation-layer circuits, a module composes those,
# and a recurrent cell composes other modules too; iverilog pulls each one in from
# the operation and module rtl/ directories, where the file name equals the module
# name. Operation directories come first, the same -y order test_guards.py builds,
# so a name present in both trees resolves to the same file whichever tool
# elaborated it.
LIBS := $(patsubst %,-y %,$(sort $(wildcard operation/*/rtl)) $(sort $(wildcard module/*/rtl)))

# Sweep target list, derived from the same rtl/ glob the -y search path uses, so a
# newly added op or module folder is swept without editing this file.
SWEEP_OPS     := $(notdir $(patsubst %/rtl,%,$(sort $(wildcard operation/*/rtl))))
SWEEP_MODULES := $(notdir $(patsubst %/rtl,%,$(sort $(wildcard module/*/rtl))))

.PHONY: test vectors compile run guards equiv floor translate sweep clean

# Every unit test runs the elaboration guards first, so a deleted or reversed
# guard turns a target red without anyone typing `make guards`.
test: guards run

vectors:
	$(PYTHON) $(LAYER)/$(UNIT)/gen/gen_$(UNIT).py

# iverilog only warns on a `define the testbench uses but the generated params
# header no longer defines: it expands to nothing, the compare still elaborates,
# and the run still prints PASS. The compile output is captured to
# build/compile.log so that warning class is a build failure, like a `^ERROR`
# line in the sim log. Three warning classes are rejected: an undefined macro,
# an implicitly defined wire, and a port width mismatch (iverilog pads or
# truncates and continues). Every RTL and testbench file here carries
# `default_nettype none, under which a typo'd net name is an elaboration error
# that fails the compile on its own, so the implicit-wire grep only bites in a
# file lacking that directive. Other iverilog warnings stay warnings.
compile: vectors
	@mkdir -p $(LAYER)/$(UNIT)/build
	@$(IVERILOG) -g2001 -Wall -I$(LAYER) $(LIBS) -o $(LAYER)/$(UNIT)/build/sim $(LAYER)/$(UNIT)/rtl/*.v $(LAYER)/$(UNIT)/tb/$(UNIT)_tb.v >$(LAYER)/$(UNIT)/build/compile.log 2>&1; \
	status=$$?; \
	cat $(LAYER)/$(UNIT)/build/compile.log; \
	[ $$status -eq 0 ] || { echo "*** $(UNIT): COMPILE FAILED ***"; exit 1; }; \
	! grep -q "undefined (and assumed null)" $(LAYER)/$(UNIT)/build/compile.log \
		|| { echo "*** $(UNIT): UNDEFINED MACRO ***"; exit 1; }; \
	! grep -q "implicit definition of wire" $(LAYER)/$(UNIT)/build/compile.log \
		|| { echo "*** $(UNIT): IMPLICIT WIRE ***"; exit 1; }; \
	! grep -qE "expects [0-9]+ bits, got [0-9]+" $(LAYER)/$(UNIT)/build/compile.log \
		|| { echo "*** $(UNIT): PORT WIDTH MISMATCH ***"; exit 1; }

# Testbenches resolve vec/<unit>.vec relative to the unit directory.
# The `vvp | tee` pipe reports tee's status, so the pipe's own exit code proves
# nothing. The two greps below are the pass/fail decision: the log must carry a
# `^PASS` line and must carry no `^ERROR` line. A vvp that dies, is missing, or
# prints nothing leaves a log with no `^PASS` and turns the target red there.
run: compile
	cd $(LAYER)/$(UNIT) && $(VVP) build/sim | tee build/sim.log
	@grep -q "^PASS" $(LAYER)/$(UNIT)/build/sim.log || { echo "*** $(UNIT): TEST FAILED ***"; exit 1; }
	@# vvp exits 0 after a runtime ERROR line, so an unreadable $$readmemb ROM
	@# would otherwise reach the PASS grep with an all-x memory.
	@! grep -q "^ERROR" $(LAYER)/$(UNIT)/build/sim.log || { echo "*** $(UNIT): SIMULATION ERROR ***"; exit 1; }
	@# $$readmemb and $$readmemh warn and keep going on a memory-load mismatch, so
	@# the two classes that corrupt the loaded table are rejected here. "Not enough
	@# words" leaves the tail of the memory at x, and "Excess binary digits" keeps
	@# only the low bits of an over-wide word, loading a value the file does not
	@# hold. "Too many words" is the accepted class: every requested entry is
	@# loaded from the file and only the surplus tail is dropped, which is how a
	@# shared table sized for the widest instance is read by a narrower one.
	@! grep -qE "Not enough words in the file|Excess binary digits" $(LAYER)/$(UNIT)/build/sim.log \
		|| { echo "*** $(UNIT): MEMORY LOAD MISMATCH ***"; exit 1; }

# Elaborate every guarded module with a violating parameter set and require the
# build to fail; unit-independent, so it takes no OP= or MODULE=.
guards:
	$(PYTHON) test_guards.py

# Walk each inlined copy against its standalone reference module, escalate the
# depths that carry a sizing parameter, and probe the dropped clamp arms;
# unit-independent, so it takes no OP= or MODULE=.
equiv:
	$(PYTHON) test_equiv.py

# Require the globbed sweep target list to match the mapping.yaml registry, so a
# deleted or renamed unit folder turns the sweep red instead of leaving it.
floor:
	$(PYTHON) sweep_floor.py "$(SWEEP_OPS)" "$(SWEEP_MODULES)"

# mapping.yaml, syn/translate.py, and the RULE_IMP.md rule 6 ledger carry no RTL
# of their own, so no unit target covers them. The whole tests/syn/ directory
# runs here, gating translation in both the reject and the accept direction and
# picking up any test later added there; it runs in the sweep.
translate:
	$(PYTHON) -m pytest -q ../../../tests/syn/

# The mapping.yaml floor, every operation target, every module target, the guards
# suite, the reference-form equivalence walk, and the translate pytest. Runs the
# whole list before reporting, so the summary names every red target, and exits
# nonzero if the list is non-empty. Each unit's own output stays in
# <layer>/<unit>/build/sim.log; the floor, guards, equiv, and translate logs stay
# in build/ and are printed when their step fails.
sweep:
	@mkdir -p build; fail=""; \
	echo "=== floor"; \
	$(MAKE) --no-print-directory floor >build/floor.log 2>&1 \
		|| { fail="$$fail floor"; cat build/floor.log; }; \
	echo "=== guards"; \
	$(MAKE) --no-print-directory guards >build/guards.log 2>&1 \
		|| { fail="$$fail guards"; cat build/guards.log; }; \
	echo "=== equiv"; \
	$(MAKE) --no-print-directory equiv >build/equiv.log 2>&1 \
		|| { fail="$$fail equiv"; cat build/equiv.log; }; \
	echo "=== translate"; \
	$(MAKE) --no-print-directory translate >build/translate.log 2>&1 \
		|| { fail="$$fail translate"; cat build/translate.log; }; \
	for op in $(SWEEP_OPS); do \
		echo "=== OP=$$op"; \
		$(MAKE) --no-print-directory run OP=$$op >/dev/null || fail="$$fail OP=$$op"; \
	done; \
	for mod in $(SWEEP_MODULES); do \
		echo "=== MODULE=$$mod"; \
		$(MAKE) --no-print-directory run MODULE=$$mod >/dev/null || fail="$$fail MODULE=$$mod"; \
	done; \
	if [ -n "$$fail" ]; then \
		echo "*** SWEEP FAILED:$$fail"; exit 1; \
	fi; \
	echo "SWEEP PASSED: $(words $(SWEEP_OPS)) operations, $(words $(SWEEP_MODULES)) modules, floor, guards, equiv, translate"

clean:
	rm -rf $(LAYER)/$(UNIT)/build $(LAYER)/$(UNIT)/vec/*.vec $(LAYER)/$(UNIT)/vec/*_params.vh $(LAYER)/$(UNIT)/vec/*.hex
