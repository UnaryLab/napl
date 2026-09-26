"""
Check every elaboration guard in the tree: the violating parameter set must make
`iverilog` fail with the guard's own module name, and the legal parameter set
right next to it must still elaborate.

Guards are zero-area `generate` blocks that instantiate an undefined module, so
nothing else in the flow notices when one is deleted or written backwards. This
file is that notice: delete a guard block and its row here turns red.

Enrolling a new guard is one row in GUARDS: the RTL module, its layer and unit
folder, the guard module name, and one violating plus one legal parameter set.

Run inside the `napl` conda env, from src/napl/imp/:
    python test_guards.py
    make guards
"""
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

IMP = Path(__file__).resolve().parent

# rtl_module, layer, unit folder, guard module name, violating params, legal params.
GUARDS = [
    # WIDTH 4 with ENTRY 8 (7 features + bias) violates 2**(WIDTH-1) > ENTRY;
    # ENTRY 7 (no bias) is the legal boundary right below it.
    ("linear_mix_unipolar", "module", "linear_mix",
     "ERROR_linear_mix_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    ("linear_mix_bipolar", "module", "linear_mix",
     "ERROR_linear_mix_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    # A 3x1 kernel over one input channel gives ENTRY 4 with the bias, which
    # WIDTH 3 cannot hold; ENTRY 3 (no bias) is the legal boundary right below it.
    ("conv_mix_unipolar", "module", "conv_mix",
     "ERROR_conv_mix_WIDTH_too_small_for_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 3, "HAS_BIAS": 0, "LANES": 1}),
    ("conv_mix_bipolar", "module", "conv_mix",
     "ERROR_conv_mix_WIDTH_too_small_for_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 3, "HAS_BIAS": 0, "LANES": 1}),
    # The same geometry produces one output position, so LANES 2 is a lane count
    # the geometry cannot fill and LANES 1 is the matching one.
    ("conv_mix_unipolar", "module", "conv_mix",
     "ERROR_conv_mix_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1}),
    ("conv_mix_bipolar", "module", "conv_mix",
     "ERROR_conv_mix_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1}),
    # conv_ugemm generates its weight and bias streams in hardware, so it carries
    # SEQ_WIDTH on top of conv's geometry; the two guards are conv's.
    ("conv_ugemm_unipolar", "module", "conv_ugemm",
     "ERROR_conv_ugemm_WIDTH_too_small_for_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 3, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 3, "SCALE": 3, "HAS_BIAS": 0, "LANES": 1}),
    ("conv_ugemm_bipolar", "module", "conv_ugemm",
     "ERROR_conv_ugemm_WIDTH_too_small_for_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 3, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 3, "SCALE": 3, "HAS_BIAS": 0, "LANES": 1}),
    # The same geometry produces one output position, so LANES 2 is a lane count
    # the geometry cannot fill and LANES 1 is the matching one.
    ("conv_ugemm_unipolar", "module", "conv_ugemm",
     "ERROR_conv_ugemm_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1}),
    ("conv_ugemm_bipolar", "module", "conv_ugemm",
     "ERROR_conv_ugemm_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1}),
    # conv_gaines composes linear_gaines over the im2col patch, so only the lane
    # count is its own guard, and the one output position the geometry produces
    # makes LANES 2 unfillable while LANES 1 matches.
    ("conv_gaines_unipolar", "module", "conv_gaines",
     "ERROR_conv_gaines_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "SCALE_WIDTH": 2, "HAS_BIAS": 1, "SCALED": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "SCALE_WIDTH": 2, "HAS_BIAS": 1, "SCALED": 1, "LANES": 1}),
    ("conv_gaines_bipolar", "module", "conv_gaines",
     "ERROR_conv_gaines_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "SCALE_WIDTH": 2, "HAS_BIAS": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "SEQ_WIDTH": 8, "SCALE_WIDTH": 2, "HAS_BIAS": 1, "LANES": 1}),
    # The scaled Gaines MUX selects one of ENTRY addends with SCALE_WIDTH bits, so
    # SCALE_WIDTH 2 cannot address ENTRY 8 while SCALE_WIDTH 3 matches it exactly.
    ("linear_gaines_unipolar", "module", "linear_gaines",
     "ERROR_linear_gaines_SCALE_WIDTH_must_match_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 2, "HAS_BIAS": 1, "SCALED": 1},
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 3, "HAS_BIAS": 1, "SCALED": 1}),
    ("linear_gaines_bipolar", "module", "linear_gaines",
     "ERROR_linear_gaines_SCALE_WIDTH_must_match_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 2,
      "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 3,
      "HAS_BIAS": 1}),
    # One addend matches a zero-bit selector span, but the scaled adder needs at
    # least two addends, so ENTRY 1 is rejected and ENTRY 2 with one select bit is
    # the legal boundary.
    ("linear_gaines_unipolar", "module", "linear_gaines",
     "ERROR_linear_gaines_SCALE_WIDTH_must_match_ENTRY",
     {"IN_FEATURES": 1, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 0, "HAS_BIAS": 0, "SCALED": 1},
     {"IN_FEATURES": 2, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 1, "HAS_BIAS": 0, "SCALED": 1}),
    ("linear_gaines_bipolar", "module", "linear_gaines",
     "ERROR_linear_gaines_SCALE_WIDTH_must_match_ENTRY",
     {"IN_FEATURES": 1, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 0, "HAS_BIAS": 0},
     {"IN_FEATURES": 2, "LANES": 1, "SEQ_WIDTH": 8, "SCALE_WIDTH": 1, "HAS_BIAS": 0}),
    ("linear_ugemm_unipolar", "module", "linear_ugemm",
     "ERROR_linear_ugemm_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    ("linear_ugemm_bipolar", "module", "linear_ugemm",
     "ERROR_linear_ugemm_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    # The guard covers both terms of 2**WIDTH + 3*ENTRY + 1 <= 2**31-1, but only
    # the WIDTH term is elaborated here since the ENTRY term needs a bus of hundreds
    # of millions of lanes and mapping.yaml's requires is what checks it.
    ("add_scale_unipolar", "operation", "add_scale",
     "ERROR_add_scale_WIDTH_and_ENTRY_overflow_32_bit_constants",
     {"SCALE": 8, "WIDTH": 31, "ENTRY": 8},
     {"SCALE": 8, "WIDTH": 30, "ENTRY": 8}),
    ("add_scale_bipolar", "operation", "add_scale",
     "ERROR_add_scale_WIDTH_and_ENTRY_overflow_32_bit_constants",
     {"SCALE": 8, "WIDTH": 31, "ENTRY": 8},
     {"SCALE": 8, "WIDTH": 30, "ENTRY": 8}),
    # The unipolar guard also rejects WIDTH 1, whose WIDTH-1 bit accumulator is
    # empty; the part-select errors that follow do not name the guard, so this row
    # stays red if the WIDTH term is dropped. WIDTH 2 is the narrowest legal one.
    ("add_scale_unipolar", "operation", "add_scale",
     "ERROR_add_scale_WIDTH_and_ENTRY_overflow_32_bit_constants",
     {"SCALE": 1, "WIDTH": 1, "ENTRY": 8},
     {"SCALE": 1, "WIDTH": 2, "ENTRY": 8}),
    # add_scale_dyn takes its scale at runtime so the guard covers SCALE_W as well.
    # Its ENTRY term is left to mapping.yaml as add_scale's is, and SCALE_W < WIDTH
    # is what the Python guard scale_max <= 2**(intwidth-1)-1 already gives every
    # translated configuration.
    ("add_scale_dyn_unipolar", "operation", "add_scale_dyn",
     "ERROR_add_scale_dyn_WIDTH_SCALE_W_and_ENTRY_overflow_32_bit_constants",
     {"SCALE_W": 2, "WIDTH": 31, "ENTRY": 8},
     {"SCALE_W": 2, "WIDTH": 30, "ENTRY": 8}),
    ("add_scale_dyn_bipolar", "operation", "add_scale_dyn",
     "ERROR_add_scale_dyn_WIDTH_SCALE_W_and_ENTRY_overflow_32_bit_constants",
     {"SCALE_W": 2, "WIDTH": 31, "ENTRY": 8},
     {"SCALE_W": 2, "WIDTH": 30, "ENTRY": 8}),
    # The largest elaboration constant is the 2*ONE_F term of C1_CODE, which the
    # 2*LO band map in clamp_sat_unipolar also reaches, so FRACWIDTH 30 pushes it to
    # 2**31, past the 32-bit signed range, and 29 is the legal boundary below.
    # clamp_sat_unipolar inherits the restriction through the instance it composes.
    ("clamp_sat_bipolar", "operation", "clamp_sat",
     "ERROR_clamp_sat_FRACWIDTH_overflows_32_bit_constants",
     {"LO": -128, "HI": 128, "FRACWIDTH": 30},
     {"LO": -128, "HI": 128, "FRACWIDTH": 29}),
    # div_scale and div_scale_dyn divide a single stream, so their accumulator bound
    # is the only 32-bit constant and WIDTH the only guarded term, and their unipolar
    # variants carry no clamp and no WIDTH so only the bipolar ones have a guard.
    ("div_scale_bipolar", "operation", "div_scale",
     "ERROR_div_scale_WIDTH_overflows_32_bit_constants",
     {"SCALE": 3, "WIDTH": 31},
     {"SCALE": 3, "WIDTH": 30}),
    ("div_scale_dyn_bipolar", "operation", "div_scale_dyn",
     "ERROR_div_scale_dyn_WIDTH_overflows_32_bit_constants",
     {"SCALE_W": 2, "WIDTH": 31},
     {"SCALE_W": 2, "WIDTH": 30}),
    # mul_scale's clamp bound 2**(WIDTH-1) is a 32-bit elaboration constant in both
    # polarities, so WIDTH 31 overflows it and WIDTH 30 is the legal boundary below.
    ("mul_scale_unipolar", "operation", "mul_scale",
     "ERROR_mul_scale_WIDTH_overflows_32_bit_constants",
     {"SCALE": 3, "WIDTH": 31},
     {"SCALE": 3, "WIDTH": 30}),
    ("mul_scale_bipolar", "operation", "mul_scale",
     "ERROR_mul_scale_WIDTH_overflows_32_bit_constants",
     {"SCALE": 3, "WIDTH": 31},
     {"SCALE": 3, "WIDTH": 30}),
    # mgu's run must outlast the decorrelation shift register, which on the
    # elaborated parameters is SEQ_WIDTH > SR_WIDTH so equal widths violate it and
    # one more SEQ_WIDTH bit is the legal boundary above it, while the gate
    # accumulator width is guarded inside linear_mix_bipolar, which mgu instantiates.
    ("mgu_hard_mix_bipolar", "module", "mgu_hard_mix",
     "ERROR_mgu_hard_mix_SEQ_WIDTH_must_exceed_SR_WIDTH",
     {"LANES": 2, "IN_SIZE": 2, "WIDTH": 8, "SEQ_WIDTH": 4, "SR_WIDTH": 4,
      "HAS_BIAS_F": 1, "HAS_BIAS_N": 1},
     {"LANES": 2, "IN_SIZE": 2, "WIDTH": 8, "SEQ_WIDTH": 5, "SR_WIDTH": 4,
      "HAS_BIAS_F": 1, "HAS_BIAS_N": 1}),
    # A non-power-of-two DEPTH, a DEPTH the index cannot span, and an empty
    # buffer are the three ways DEPTH != 2**WIDTH arises; DEPTH 1 with WIDTH 0
    # matches 2**WIDTH but leaves a zero-bit index, which only the DEPTH < 2 term
    # rejects, so its part-select errors cannot stand in for the guard.
    ("div_cordiv", "operation", "div_cordiv",
     "ERROR_div_cordiv_DEPTH_must_equal_two_to_the_WIDTH",
     {"DEPTH": 6, "WIDTH": 3},
     {"DEPTH": 8, "WIDTH": 3}),
    ("div_cordiv", "operation", "div_cordiv",
     "ERROR_div_cordiv_DEPTH_must_equal_two_to_the_WIDTH",
     {"DEPTH": 8, "WIDTH": 2},
     {"DEPTH": 4, "WIDTH": 2}),
    ("div_cordiv", "operation", "div_cordiv",
     "ERROR_div_cordiv_DEPTH_must_equal_two_to_the_WIDTH",
     {"DEPTH": 0, "WIDTH": 1},
     {"DEPTH": 2, "WIDTH": 1}),
    ("div_cordiv", "operation", "div_cordiv",
     "ERROR_div_cordiv_DEPTH_must_equal_two_to_the_WIDTH",
     {"DEPTH": 1, "WIDTH": 0},
     {"DEPTH": 2, "WIDTH": 1}),
    # The scaled Gaines adder steps a SELECT_WIDTH counter through an ENTRY-deep
    # ROM so the two spans must be equal, ENTRY 5 is not the span of SELECT_WIDTH 3
    # while ENTRY 8 is, and the unscaled arm carries neither so only the scaled one
    # has a guard.
    ("add_gaines", "operation", "add_gaines",
     "ERROR_add_gaines_ENTRY_must_equal_two_to_the_SELECT_WIDTH",
     {"SCALED": 1, "ENTRY": 5, "SELECT_WIDTH": 3},
     {"SCALED": 1, "ENTRY": 8, "SELECT_WIDTH": 3}),
    # The sequence value is left-aligned onto the input's fraction grid, so the
    # input grid must be at least as fine: FRAC 7 is coarser than WIDTH 8 while
    # FRAC 8 is the equal-grid boundary right above it.
    ("encode", "operation", "encode",
     "ERROR_encode_FRAC_too_small_for_WIDTH",
     {"WIDTH": 8, "FRAC": 7},
     {"WIDTH": 8, "FRAC": 8}),
    # GENERATOR selects one sequence circuit; 4 names none of the four branches
    # while 3, temporalseq, is the last one that does.
    ("encode", "operation", "encode",
     "ERROR_encode_GENERATOR_must_select_a_supported_sequence",
     {"WIDTH": 8, "FRAC": 9, "GENERATOR": 4},
     {"WIDTH": 8, "FRAC": 9, "GENERATOR": 3}),
    # A zero-bit sequence has no points, so WIDTH 0 violates the bound while
    # WIDTH 1, which walks its two-entry table, is the narrowest legal one.
    ("sobol", "operation", "encode",
     "ERROR_sobol_WIDTH_must_be_at_least_1",
     {"WIDTH": 0},
     {"WIDTH": 1}),
    # get_lfsr_seq raises below width 2, where there is no feedback polynomial, so
    # WIDTH 1 violates the bound and WIDTH 2 is the narrowest legal one. Both LFSR
    # forms carry it: lfsr_ext extends the same maximal cycle.
    ("lfsr", "operation", "encode",
     "ERROR_lfsr_WIDTH_must_be_at_least_2",
     {"WIDTH": 1},
     {"WIDTH": 2}),
    ("lfsr_ext", "operation", "encode",
     "ERROR_lfsr_ext_WIDTH_must_be_at_least_2",
     {"WIDTH": 1},
     {"WIDTH": 2}),
    # A zero-bit ramp has no points, so WIDTH 0 violates the bound while WIDTH 1,
    # the two-point ramp, is the narrowest legal one.
    ("temporalseq", "operation", "encode",
     "ERROR_temporalseq_WIDTH_must_be_at_least_1",
     {"WIDTH": 0},
     {"WIDTH": 1}),
    # uni2bi's Python model rejects a width whose accumulator maximum
    # 2^(WIDTH-1)-1 is below the emission threshold 2, so WIDTH 2 violates the
    # bound and WIDTH 3 is the narrowest legal one.
    ("uni2bi", "operation", "uni2bi",
     "ERROR_uni2bi_WIDTH_must_be_at_least_3",
     {"WIDTH": 2},
     {"WIDTH": 3}),
]


def elaborate(rtl_module, layer, unit, parameters, work):
    """Elaborate one parameter set; return (exit status, combined output)."""
    overrides = ", ".join(f".{name} ({value})" for name, value in parameters.items())
    top = work / "guard_top.v"
    top.write_text(
        "`timescale 1ns/1ps\n"
        "module guard_top;\n"
        f"    {rtl_module} #({overrides}) u_dut ();\n"
        "endmodule\n"
    )
    command = ["iverilog", "-g2001", "-Wall", f"-I{layer}"]
    # Operation directories precede module ones, the same -y order the Makefile
    # passes, so a name in both trees resolves to the same file whichever tool
    # elaborated it.
    for library in sorted(glob("operation/*/rtl", root_dir=IMP)) + sorted(glob("module/*/rtl", root_dir=IMP)):
        command += ["-y", library]
    command += ["-o", str(work / "sim")]
    command += sorted(glob(f"{layer}/{unit}/rtl/*.v", root_dir=IMP)) + [str(top)]
    done = subprocess.run(command, capture_output=True, text=True, cwd=IMP)
    return done.returncode, done.stdout + done.stderr


def test_guards_reject_violating_parameters():
    """Each elaboration guard fails its violating parameter set and passes a legal one."""
    failures = []
    with tempfile.TemporaryDirectory() as name:
        work = Path(name)
        for rtl_module, layer, unit, guard, bad, good in GUARDS:
            before = len(failures)
            status, output = elaborate(rtl_module, layer, unit, bad, work)
            if status == 0:
                failures.append(f"{rtl_module} {bad}: elaborated, guard {guard} did not fire")
            elif guard not in output:
                failures.append(f"{rtl_module} {bad}: failed without naming {guard}:\n{output}")
            status, output = elaborate(rtl_module, layer, unit, good, work)
            if status != 0:
                failures.append(f"{rtl_module} {good}: legal parameters failed:\n{output}")
            verdict = "ok" if len(failures) == before else "FAILED"
            print(f"  {rtl_module}: {guard} fires on {bad}, silent on {good} -- {verdict}")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_guards_reject_violating_parameters()
    print("Test passed.")
    sys.exit(0)
