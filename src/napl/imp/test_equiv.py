"""
Host-vs-standalone reference-form equivalence, by RANDOM WALK. NOT A PROOF.

Several units inline another operation's logic into their own outer body while
that operation also exists as a standalone module with its own tests. This file
drives both forms from one stimulus and fails on the first cycle they disagree.

Three mechanisms live here, each independently gated:

1. Random walk (PAIRS). Every pair runs WALK_CYCLES random cycles including
   randomly asserted resets. Coverage ceiling: a walk visits the states its
   stimulus happens to reach, so a divergence reachable only on a rare state
   sequence can pass this check. It bounds nothing and proves nothing about
   states the walk never entered.
2. Depth escalation (SIZING). A pair whose inline copy carries a sizing
   parameter also runs at depths GREATER than the depth its generator commits,
   because a shift path that only exists above the committed depth is elaborated
   nowhere else in the flow. This always runs; it is not a flag.
3. Reachability probe (PROBES). A copy that drops a clamp arm its Python
   reference model applies is driven over EVERY input sequence of length
   PROBE_SEQ_LEN from reset, and the pre-clamp sum it tapped must stay inside the
   bounds the dropped arm would have enforced. Ceiling: exhaustive over sequences
   of that length, not over longer ones.

Coverage dropped on purpose, printed by every run:
  - Only square_dff carries a sizing parameter on the inline copy. Every other
    pair inlines a fixed size, so mechanism 2 has nothing to escalate there.
  - PAIRS is hand-maintained and has no floor against mapping.yaml, so a pair
    deleted from the table leaves no trace. Each pair does state how many nets it
    compares, so a pair that stays in the table but stops comparing is red.

Run inside the `napl` conda env, from src/napl/imp/:
    python test_equiv.py
    make equiv
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

IMP = Path(__file__).resolve().parent

WALK_CYCLES = 200000
WALK_SEED = 20260811
# One cycle in RESET_PERIOD asserts i_rst_n low, so a walk covers reset from
# dirtied state, not only the opening reset.
RESET_PERIOD = 512
PROBE_SEQ_LEN = 16

# host module, its layer/unit folder, host input ports, host output ports,
# reference module, its layer/unit folder, reference port -> host-side expression,
# and the reference outputs checked against host internal nets.
PAIRS = [
    dict(name="square_dff_unipolar inline dff", host="square_dff_unipolar",
         host_unit=("operation", "square_dff"), inputs=["i_input"], outputs=["o_output"],
         ref="dff", ref_unit=("operation", "dff"),
         ref_inputs={"i_input": "r_i_input"},
         checks=[("o_output", "host.in_d[0]")],
         sizing=dict(param="DEPTH", committed=1, deeper=[2, 3, 5],
                     generator="operation/square_dff/gen/gen_square_dff.py",
                     literal='"depth": 1')),
    dict(name="square_dff_bipolar inline dff", host="square_dff_bipolar",
         host_unit=("operation", "square_dff"), inputs=["i_input"], outputs=["o_output"],
         ref="dff", ref_unit=("operation", "dff"),
         ref_inputs={"i_input": "r_i_input"},
         checks=[("o_output", "host.in_d[0]")],
         sizing=dict(param="DEPTH", committed=1, deeper=[2, 3, 5],
                     generator="operation/square_dff/gen/gen_square_dff.py",
                     literal='"depth": 1')),
    dict(name="lt_rc inline sync_skewed", host="lt_rc", host_unit=("operation", "lt_rc"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output"],
         ref="sync_skewed", ref_unit=("operation", "sync_skewed"), ref_params={"WIDTH": 2},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.sync_0"), ("o_output_1", "host.sync_1")]),
    dict(name="gt_rc inline sync_skewed", host="gt_rc", host_unit=("operation", "gt_rc"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output"],
         ref="sync_skewed", ref_unit=("operation", "sync_skewed"), ref_params={"WIDTH": 2},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.sync_0"), ("o_output_1", "host.sync_1")]),
    dict(name="min_rc inline sync_skewed", host="min_rc", host_unit=("operation", "min_rc"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output", "o_index"],
         ref="sync_skewed", ref_unit=("operation", "sync_skewed"), ref_params={"WIDTH": 2},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.sync_0"), ("o_output_1", "host.sync_1")]),
    dict(name="max_rc inline sync_skewed", host="max_rc", host_unit=("operation", "max_rc"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output", "o_index"],
         ref="sync_skewed", ref_unit=("operation", "sync_skewed"), ref_params={"WIDTH": 2},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.sync_0"), ("o_output_1", "host.sync_1")]),
    dict(name="min_sync inline sync", host="min_sync", host_unit=("operation", "min_sync"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output"],
         ref="sync", ref_unit=("operation", "sync"), ref_params={"DEPTH": 1},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.sync_0"), ("o_output_1", "host.sync_1")]),
    dict(name="max_sync inline sync", host="max_sync", host_unit=("operation", "max_sync"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output"],
         ref="sync", ref_unit=("operation", "sync"), ref_params={"DEPTH": 1},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.sync_0"), ("o_output_1", "host.sync_1")]),
    dict(name="add_desync inline desync", host="add_desync",
         host_unit=("operation", "add_desync"),
         inputs=["i_input_0", "i_input_1"], outputs=["o_output"],
         ref="desync", ref_unit=("operation", "desync"), ref_params={"DEPTH": 1},
         ref_inputs={"i_input_0": "r_i_input_0", "i_input_1": "r_i_input_1"},
         checks=[("o_output_0", "host.desync_0"), ("o_output_1", "host.desync_1")]),
    dict(name="sqrt_tracejkff_unipolar inline jkff", host="sqrt_tracejkff_unipolar",
         host_unit=("operation", "sqrt_tracejkff"), inputs=["i_input"], outputs=["o_output"],
         ref="jkff", ref_unit=("operation", "jkff"),
         # The host carries the K=1 form of the recurrence, trace <= ~trace &
         # shuffled, which is the jkff characteristic equation at that constant,
         # so the reference is driven with K tied to 1.
         ref_inputs={"i_input_j": "host.shuffled", "i_input_k": "1'b1"},
         checks=[("o_q", "host.trace")]),
    dict(name="div_iscb_unipolar inline sync_skewed", host="div_iscb_unipolar",
         host_unit=("operation", "div_iscb"), inputs=["i_dividend", "i_divisor"],
         outputs=["o_output"],
         ref="sync_skewed", ref_unit=("operation", "sync_skewed"), ref_params={"WIDTH": 3},
         ref_inputs={"i_input_0": "r_i_dividend", "i_input_1": "r_i_divisor"},
         checks=[("o_output_0", "host.sync_out"), ("o_output_1", "host.sync_divisor")]),
    dict(name="div_iscb_unipolar inline div_cordiv", host="div_iscb_unipolar",
         host_unit=("operation", "div_iscb"), inputs=["i_dividend", "i_divisor"],
         outputs=["o_output"],
         ref="div_cordiv", ref_unit=("operation", "div_cordiv"),
         ref_params={"DEPTH": 2, "WIDTH": 1},
         ref_inputs={"i_dividend": "host.sync_out", "i_divisor": "host.sync_divisor"},
         checks=[("o_output", "host.quotient")]),
    dict(name="sqrt_traceiscb_unipolar inline div_cordiv", host="sqrt_traceiscb_unipolar",
         host_unit=("operation", "sqrt_traceiscb"), inputs=["i_input"], outputs=["o_output"],
         ref="div_cordiv", ref_unit=("operation", "div_cordiv"),
         ref_params={"DEPTH": 2, "WIDTH": 1},
         ref_inputs={"i_dividend": "host.dividend", "i_divisor": "host.divisor"},
         checks=[("o_output", "host.quotient")]),
    dict(name="sqrt_traceiscb_bipolar inline div_cordiv", host="sqrt_traceiscb_bipolar",
         host_unit=("operation", "sqrt_traceiscb"), inputs=["i_input"], outputs=["o_output"],
         ref="div_cordiv", ref_unit=("operation", "div_cordiv"),
         ref_params={"DEPTH": 2, "WIDTH": 1},
         ref_inputs={"i_dividend": "host.dividend", "i_divisor": "host.divisor"},
         checks=[("o_output", "host.quotient")]),
    dict(name="sqrt_traceiscb_bipolar inline bi2uni", host="sqrt_traceiscb_bipolar",
         host_unit=("operation", "sqrt_traceiscb"), inputs=["i_input"], outputs=["o_output"],
         ref="bi2uni", ref_unit=("operation", "bi2uni"), ref_params={"WIDTH": 3},
         ref_inputs={"i_input": "host.output_bit"},
         checks=[("o_output", "host.out_bit")]),
    dict(name="sqrt_tracejkff_bipolar inline bi2uni", host="sqrt_tracejkff_bipolar",
         host_unit=("operation", "sqrt_tracejkff"), inputs=["i_input"], outputs=["o_output"],
         ref="bi2uni", ref_unit=("operation", "bi2uni"), ref_params={"WIDTH": 2},
         ref_inputs={"i_input": "host.o_output"},
         checks=[("o_output", "host.out_uni")]),
    # div_iscb_bipolar needs no pair here: it instantiates the standalone bi2uni,
    # uni2bi, and signabs at WIDTH 3 rather than carrying copies of them.
    dict(name="sigmoid_hard copy of uni2bi", host="sigmoid_hard",
         host_unit=("operation", "sigmoid_hard"), inputs=["i_input"], outputs=["o_output"],
         ref="uni2bi", ref_unit=("operation", "uni2bi"), ref_params={"WIDTH": 3},
         ref_inputs={"i_input": "r_i_input"},
         checks=[("o_output", "host.o_output")]),
]

# module, its layer/unit folder, parameters, the pre-clamp sum net, and the bounds
# the dropped clamp arms would have enforced, where a None bound is a clamp the
# module still carries and so is not probed.
PROBES = [
    dict(name="uni2bi drops both clamp arms", module="uni2bi",
         unit=("operation", "uni2bi"), params={"WIDTH": 3},
         sum_net="dut.sum", lo=-4, hi=3),
    dict(name="sigmoid_hard drops both clamp arms", module="sigmoid_hard",
         unit=("operation", "sigmoid_hard"), params={},
         sum_net="dut.acc_sum", lo=-4, hi=3),
    dict(name="relu_sat sub stage drops the upper clamp arm", module="relu_sat",
         unit=("operation", "relu_sat"), params={},
         sum_net="dut.sum_sub", lo=None, hi=2),
    dict(name="relu_sat add stage drops both clamp arms", module="relu_sat",
         unit=("operation", "relu_sat"), params={},
         sum_net="dut.sum_add", lo=0, hi=4),
]


def override(params):
    """Render a parameter map as a Verilog instance override list."""
    return " #(" + ", ".join(f".{k}({v})" for k, v in params.items()) + ")" if params else ""


def walk_tb(pair, host_params, ref_params):
    """Emit the testbench that walks one pair's two forms from identical stimulus."""
    inputs = pair["inputs"]
    declare = "".join(f"    reg r_{name};\n" for name in inputs)
    declare += "".join(f"    wire w_{name};\n" for name in pair["outputs"])
    declare += "".join(f"    wire ref_{name};\n" for name, _ in pair["checks"])
    host_ports = ", ".join([".i_clk(i_clk)", ".i_rst_n(i_rst_n)"]
                           + [f".{n}(r_{n})" for n in inputs]
                           + [f".{n}(w_{n})" for n in pair["outputs"]])
    ref_ports = ", ".join([".i_clk(i_clk)", ".i_rst_n(i_rst_n)"]
                          + [f".{p}({e})" for p, e in pair["ref_inputs"].items()]
                          + [f".{n}(ref_{n})" for n, _ in pair["checks"]])
    drive = "".join(f"            r_{name} = $random(seed);\n" for name in inputs)
    checks = ""
    for name, host_net in pair["checks"]:
        # An x on either net is counted too: x !== x is false, so a pair whose
        # nets never resolve would otherwise report a green that compares nothing.
        checks += (
            f"            if (ref_{name} !== {host_net} || ref_{name} === 1'bx\n"
            f"                    || {host_net} === 1'bx) begin\n"
            "                mismatches = mismatches + 1;\n"
            "                if (first_bad < 0) begin\n"
            "                    first_bad = cycle;\n"
            f'                    $display("DIVERGENCE cycle=%0d check={name} vs {host_net} '
            f'ref=%b host=%b", cycle, ref_{name}, {host_net});\n'
            "                end\n"
            "            end\n"
        )
    return f"""`timescale 1ns/1ps
module equiv_top;
    integer cycle;
    integer mismatches;
    integer first_bad;
    integer seed;
    reg i_clk;
    reg i_rst_n;
{declare}
    {pair["host"]}{override(host_params)} host ({host_ports});
    {pair["ref"]}{override(ref_params)} refm ({ref_ports});

    always #5 i_clk = ~i_clk;

    initial begin
        i_clk = 1'b0;
        i_rst_n = 1'b1;
        mismatches = 0;
        first_bad = -1;
        seed = {WALK_SEED};
{"".join(f"        r_{name} = 1'b0;" + chr(10) for name in inputs)}        i_rst_n = 1'b0;
        @(negedge i_clk);
        @(negedge i_clk);
        i_rst_n = 1'b1;
        for (cycle = 0; cycle < {WALK_CYCLES}; cycle = cycle + 1) begin
            @(negedge i_clk);
            i_rst_n = ((cycle % {RESET_PERIOD}) == ({RESET_PERIOD} - 1)) ? 1'b0 : 1'b1;
{drive}            #1;
{checks}        end
        $display("RESULT mismatches=%0d first=%0d cycles=%0d", mismatches, first_bad, cycle);
        $finish;
    end
endmodule
"""


def probe_tb(probe):
    """Emit the testbench that walks every input sequence of PROBE_SEQ_LEN from reset."""
    lo, hi = probe["lo"], probe["hi"]
    tests = []
    if lo is not None:
        tests.append(f"sum_now < {lo}")
    if hi is not None:
        tests.append(f"sum_now > {hi}")
    bound = " || ".join(tests)
    return f"""`timescale 1ns/1ps
module probe_top;
    integer cycle;
    integer seq;
    integer step;
    integer violations;
    integer sum_now;
    integer sum_lo;
    integer sum_hi;
    reg i_clk;
    reg i_rst_n;
    reg [{PROBE_SEQ_LEN - 1}:0] bits;
    wire w_out;

    {probe["module"]}{override(probe["params"])} dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(bits[0]), .o_output(w_out));

    always #5 i_clk = ~i_clk;

    initial begin
        i_clk = 1'b0;
        i_rst_n = 1'b1;
        bits = 0;
        cycle = 0;
        violations = 0;
        sum_lo = 0;
        sum_hi = 0;
        for (seq = 0; seq < {1 << PROBE_SEQ_LEN}; seq = seq + 1) begin
            i_rst_n = 1'b0;
            @(negedge i_clk);
            i_rst_n = 1'b1;
            for (step = 0; step < {PROBE_SEQ_LEN}; step = step + 1) begin
                @(negedge i_clk);
                bits = seq >> step;
                #1;
                sum_now = {probe["sum_net"]};
                if (cycle == 0) begin
                    sum_lo = sum_now;
                    sum_hi = sum_now;
                end
                if (sum_now < sum_lo) sum_lo = sum_now;
                if (sum_now > sum_hi) sum_hi = sum_now;
                if ({bound}) begin
                    violations = violations + 1;
                    if (violations == 1)
                        $display("REACHED cycle=%0d seq=%0d step=%0d sum=%0d",
                                 cycle, seq, step, sum_now);
                end
                cycle = cycle + 1;
            end
        end
        $display("RESULT violations=%0d min=%0d max=%0d cycles=%0d",
                 violations, sum_lo, sum_hi, cycle);
        $finish;
    end
endmodule
"""


def simulate(text, top, sources, work, unit):
    """Compile one generated testbench with its sources and return the vvp output.

    vvp runs in the unit's own folder, the cwd each op's own testbench is given,
    so a `$readmemb("vec/...")` inside the design resolves to that unit's ROM.

    Each call compiles into its own scratch subdir so the source and `-o` binary
    are unique per pair; concurrent compiles cannot clobber a binary another run
    is still executing.
    """
    scratch = Path(tempfile.mkdtemp(dir=work))
    path = scratch / f"{top}.v"
    path.write_text(text)
    build = scratch / "sim"
    # Operation directories precede module ones, the same -y order the Makefile
    # passes, so a circuit a host instantiates resolves to the same file here.
    libraries = []
    for library in sorted(IMP.glob("operation/*/rtl")) + sorted(IMP.glob("module/*/rtl")):
        libraries += ["-y", str(library.relative_to(IMP))]
    done = subprocess.run(["iverilog", "-g2001"] + libraries + ["-o", str(build)]
                          + sources + [str(path)],
                          capture_output=True, text=True, cwd=IMP)
    if done.returncode != 0:
        return "COMPILE FAILED\n" + done.stdout + done.stderr
    done = subprocess.run(["vvp", str(build)], capture_output=True, text=True,
                          cwd=IMP / unit[0] / unit[1])
    return done.stdout + done.stderr


def sources_for(*units):
    """Every RTL file of the named layer/unit folders, deduplicated in order."""
    files = []
    for layer, unit in units:
        for path in sorted((IMP / layer / unit / "rtl").glob("*.v")):
            name = str(path.relative_to(IMP))
            if name not in files:
                files.append(name)
    return files


def param_sets(pair):
    """Yield (label, host params, reference params) including depths above committed."""
    sizing = pair.get("sizing")
    if not sizing:
        yield "fixed", {}, pair.get("ref_params", {})
        return
    name = sizing["param"]
    source = (IMP / sizing["generator"]).read_text()
    assert sizing["literal"] in source, (
        f"{sizing['generator']} no longer commits {sizing['literal']}; "
        f"{pair['name']} would stop running deeper than committed")
    assert min(sizing["deeper"]) > sizing["committed"], "deeper depths must exceed committed"
    for value in [sizing["committed"]] + sizing["deeper"]:
        tag = "committed" if value == sizing["committed"] else "DEEPER"
        yield f"{name}={value} ({tag})", {name: value}, {name: value}


def test_host_and_reference_forms_agree():
    """Each inlined copy matches its standalone reference module over a random walk."""
    failures = []
    print(f"random walk: {WALK_CYCLES} cycles per run, reset every {RESET_PERIOD}, "
          f"seed {WALK_SEED}. This is a walk, not a proof.")
    with tempfile.TemporaryDirectory() as name:
        work = Path(name)
        for pair in PAIRS:
            nets = len(pair["checks"])
            if not nets:
                failures.append(f"{pair['name']}: compares no nets, so a run of it "
                                "reports nothing about the two forms")
                print(f"  {pair['name']} -- FAILED (0 nets compared)")
                continue
            sources = sources_for(pair["host_unit"], pair["ref_unit"])
            for label, host_params, ref_params in param_sets(pair):
                output = simulate(walk_tb(pair, host_params, ref_params),
                                  "equiv_top", sources, work, pair["host_unit"])
                found = re.search(r"RESULT mismatches=(\d+) first=(-?\d+)", output)
                if not found:
                    failures.append(f"{pair['name']} [{label}]: no result:\n{output}")
                    print(f"  {pair['name']} [{label}] -- FAILED (no result)")
                    continue
                count, first = int(found.group(1)), int(found.group(2))
                if count:
                    failures.append(f"{pair['name']} [{label}]: {count} mismatches, "
                                    f"first at cycle {first}")
                verdict = "ok" if count == 0 else f"FAILED {count} mismatches, first cycle {first}"
                print(f"  {pair['name']} [{label}]: {nets} nets x {WALK_CYCLES} cycles "
                      f"-- {verdict}")
    assert not failures, "\n".join(failures)


def test_dropped_clamp_arms_stay_unreachable():
    """Each copy that drops a clamp arm keeps its sum inside that arm's bound."""
    failures = []
    print(f"reachability probe: every input sequence of length {PROBE_SEQ_LEN} from reset. "
          "Exhaustive at that length only.")
    with tempfile.TemporaryDirectory() as name:
        work = Path(name)
        for probe in PROBES:
            output = simulate(probe_tb(probe), "probe_top", sources_for(probe["unit"]),
                              work, probe["unit"])
            found = re.search(r"RESULT violations=(\d+) min=(-?\d+) max=(-?\d+)", output)
            if not found:
                failures.append(f"{probe['name']}: no result:\n{output}")
                print(f"  {probe['name']} -- FAILED (no result)")
                continue
            count, low, high = (int(found.group(index)) for index in (1, 2, 3))
            bound = f"[{'none' if probe['lo'] is None else probe['lo']}, " \
                    f"{'none' if probe['hi'] is None else probe['hi']}]"
            if count:
                first = re.search(r"REACHED cycle=(\d+) seq=(\d+) step=(\d+) sum=(-?\d+)", output)
                where = first.group(0) if first else "first-reach line missing"
                failures.append(f"{probe['name']}: {count} cycles left the bound "
                                f"{bound}, observed [{low}, {high}]; {where}")
            verdict = "ok" if count == 0 else f"FAILED {count} cycles out of bound"
            print(f"  {probe['name']}: sum in [{low}, {high}], bound {bound} -- {verdict}")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    print("NOT A PROOF: a random walk can miss a divergence reachable only on a "
          "rare state sequence.")
    print("Dropped coverage: PAIRS has no floor against mapping.yaml; only "
          "square_dff escalates depth.")
    # Both mechanisms run whatever the other reports, so a divergence in one does
    # not hide the state of the other.
    reports = []
    for check in (test_host_and_reference_forms_agree, test_dropped_clamp_arms_stay_unreachable):
        try:
            check()
        except AssertionError as detail:
            reports.append(f"{check.__name__}:\n{detail}")
    if reports:
        print("\n".join(reports))
        sys.exit(1)
    print("Test passed.")
    sys.exit(0)
