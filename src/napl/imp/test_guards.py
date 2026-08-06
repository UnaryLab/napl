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
    ("avgpool2d", "module", "avgpool2d",
     "ERROR_avgpool2d_DIVISOR_must_be_at_least_KERNEL_AREA",
     {"KERNEL_AREA": 4, "DIVISOR": 2, "LANES": 1},
     {"KERNEL_AREA": 4, "DIVISOR": 4, "LANES": 1}),
    # WIDTH 4 with ENTRY 8 (7 features + bias) violates 2**(WIDTH-1) > ENTRY;
    # ENTRY 7 (no bias) is the legal boundary right below it.
    ("linear_unipolar", "module", "linear",
     "ERROR_linear_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    ("linear_bipolar", "module", "linear",
     "ERROR_linear_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    # A 3x1 kernel over one input channel gives ENTRY 4 with the bias, which
    # WIDTH 3 cannot hold; ENTRY 3 (no bias) is the legal boundary right below it.
    ("conv_unipolar", "module", "conv",
     "ERROR_conv_WIDTH_too_small_for_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 3, "HAS_BIAS": 0, "LANES": 1}),
    ("conv_bipolar", "module", "conv",
     "ERROR_conv_WIDTH_too_small_for_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 3, "SCALE": 3, "HAS_BIAS": 0, "LANES": 1}),
    # The same geometry produces one output position, so LANES 2 is a lane count
    # the geometry cannot fill and LANES 1 is the matching one.
    ("conv_unipolar", "module", "conv",
     "ERROR_conv_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "WIDTH": 4, "SCALE": 4, "HAS_BIAS": 1, "LANES": 1}),
    ("conv_bipolar", "module", "conv",
     "ERROR_conv_LANES_must_equal_OUTPUT_POSITIONS",
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
    # conv_pc publishes a count bus, so its lane width is a port width and has to
    # be a parameter: COUNT_W 2 truncates the ENTRY 4 count that COUNT_W 3 holds.
    ("conv_pc_unipolar", "module", "conv_pc",
     "ERROR_conv_pc_COUNT_W_must_equal_CLOG2_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 2, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 3, "LANES": 1}),
    ("conv_pc_bipolar", "module", "conv_pc",
     "ERROR_conv_pc_COUNT_W_must_equal_CLOG2_ENTRY",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 2, "LANES": 1},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 3, "LANES": 1}),
    # The same geometry produces one output position, so LANES 2 is a lane count
    # the geometry cannot fill and LANES 1 is the matching one.
    ("conv_pc_unipolar", "module", "conv_pc",
     "ERROR_conv_pc_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 3, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 3, "LANES": 1}),
    ("conv_pc_bipolar", "module", "conv_pc",
     "ERROR_conv_pc_LANES_must_equal_OUTPUT_POSITIONS",
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 3, "LANES": 2},
     {"BATCH": 1, "IN_CHANNELS": 1, "IN_H": 3, "IN_W": 1, "OUT_CHANNELS": 1,
      "KERNEL_H": 3, "KERNEL_W": 1, "STRIDE": 1, "PADDING": 0, "DILATION": 1,
      "HAS_BIAS": 1, "COUNT_W": 3, "LANES": 1}),
    # linear_pc publishes a count bus, so its lane width is a port width and has
    # to be a parameter: COUNT_W 3 truncates the ENTRY 8 count that COUNT_W 4 holds.
    ("linear_pc_unipolar", "module", "linear_pc",
     "ERROR_linear_pc_COUNT_W_must_equal_CLOG2_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "HAS_BIAS": 1, "COUNT_W": 3},
     {"IN_FEATURES": 7, "LANES": 1, "HAS_BIAS": 1, "COUNT_W": 4}),
    ("linear_pc_bipolar", "module", "linear_pc",
     "ERROR_linear_pc_COUNT_W_must_equal_CLOG2_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "HAS_BIAS": 1, "COUNT_W": 3},
     {"IN_FEATURES": 7, "LANES": 1, "HAS_BIAS": 1, "COUNT_W": 4}),
    ("linear_ugemm_unipolar", "module", "linear_ugemm",
     "ERROR_linear_ugemm_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    ("linear_ugemm_bipolar", "module", "linear_ugemm",
     "ERROR_linear_ugemm_WIDTH_too_small_for_ENTRY",
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 8, "HAS_BIAS": 1},
     {"IN_FEATURES": 7, "LANES": 1, "SEQ_WIDTH": 8, "WIDTH": 4, "SCALE": 7, "HAS_BIAS": 0}),
    # The guard covers both terms of 2**WIDTH + 3*ENTRY + 1 <= 2**31-1. Only the
    # WIDTH term is elaborated here: the ENTRY term needs a bus of hundreds of
    # millions of lanes, so mapping.yaml's requires is what checks it.
    ("add_any_unipolar", "operation", "add_any",
     "ERROR_add_any_WIDTH_and_ENTRY_overflow_32_bit_constants",
     {"SCALE": 8, "WIDTH": 31, "ENTRY": 8},
     {"SCALE": 8, "WIDTH": 30, "ENTRY": 8}),
    ("add_any_bipolar", "operation", "add_any",
     "ERROR_add_any_WIDTH_and_ENTRY_overflow_32_bit_constants",
     {"SCALE": 8, "WIDTH": 31, "ENTRY": 8},
     {"SCALE": 8, "WIDTH": 30, "ENTRY": 8}),
    # A non-power-of-two DEPTH, a DEPTH the index cannot span, and an empty
    # buffer are the three ways DEPTH != 2**WIDTH arises.
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
    if layer == "module":
        for library in sorted(glob("operation/*/rtl")):
            command += ["-y", library]
    command += ["-o", str(work / "sim")]
    command += sorted(glob(f"{layer}/{unit}/rtl/*.v")) + [str(top)]
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
