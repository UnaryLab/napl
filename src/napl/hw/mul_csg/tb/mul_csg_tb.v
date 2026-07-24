`timescale 1ns/1ps
`default_nettype none
// GEN_WIDTH is emitted by gen/gen_mul_csg.py from the op's sizing config, so the
// DUT WIDTH parameter is inherited from the Python model. iverilog resolves this
// include relative to the compile cwd (hw/), not ../vec.
`include "mul_csg/vec/mul_csg_params.vh"
//==============================================================================
// Self-checking testbench for mul_csg (unipolar + bipolar).
//
// Reads golden vectors produced by gen/gen_mul_csg.py (from the napl Python
// model, config timestep=256 generator='sobol') and asserts both polarity RTL
// modules reproduce them cycle for cycle. Prints "PASS ..." iff every vector
// matches; the Makefile greps for that line to decide the exit status.
//
// Vector columns:  rst in_0 in_1u out_uni in_1b out_bi
//   rst==1 marks the first cycle of an independent sequence: we pulse i_rst_n
//   low so the ROM-index counters reload to 0 (matching the model's reset()).
//
// Timing: the model output at cycle t uses the counter state BEFORE its t-th
// increment. In the RTL the output is combinational in the current seq_idx
// register, so per cycle we (1) drive inputs, (2) let the output settle, (3)
// check, then (4) pulse a clock edge to advance the counters for cycle t+1.
//
// Run (from src/napl/hw/):  make test OP=mul_csg
//==============================================================================
module mul_csg_tb;
    reg                 i_clk;
    reg                 i_rst_n;
    reg                 i_in_0;
    reg  [`GEN_WIDTH:0] i_in_1u;   // operand bus = WIDTH+1 bits
    reg  [`GEN_WIDTH:0] i_in_1b;
    wire                o_out_uni;
    wire                o_out_bi;

    mul_csg_unipolar #(.WIDTH(`GEN_WIDTH)) dut_u (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_in_0  (i_in_0),
        .i_in_1  (i_in_1u),
        .o_out   (o_out_uni)
    );

    mul_csg_bipolar #(.WIDTH(`GEN_WIDTH)) dut_b (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_in_0  (i_in_0),
        .i_in_1  (i_in_1b),
        .o_out   (o_out_bi)
    );

    integer fd, code, n, fails;
    reg          rst;
    reg          a;
    integer      in1u, out_u, in1b, out_b;
    reg [1023:0] hdr_line;

    initial begin
        i_clk   = 1'b0;
        i_rst_n = 1'b1;
        i_in_0  = 1'b0;
        i_in_1u = {(`GEN_WIDTH+1){1'b0}};
        i_in_1b = {(`GEN_WIDTH+1){1'b0}};

        fd = $fopen("vec/mul_csg.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_csg.vec (run `make vectors` first)");
            $finish;
        end

        // skip the header line
        code = $fgets(hdr_line, fd);

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%d %d %d %d %d %d\n", rst, a, in1u, out_u, in1b, out_b);
            if (code == 6) begin
                // reset boundary: reload the index counters to 0 before this cycle
                if (rst == 1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                // drive inputs and let the combinational output settle
                i_in_0  = a;
                i_in_1u = in1u[`GEN_WIDTH:0];
                i_in_1b = in1b[`GEN_WIDTH:0];
                #1;

                n = n + 1;
                if (o_out_uni !== out_u[0]) begin
                    $display("FAIL uni n=%0d in_0=%b in_1=%0d : got %b exp %0d",
                             n, a, in1u, o_out_uni, out_u);
                    fails = fails + 1;
                end
                if (o_out_bi !== out_b[0]) begin
                    $display("FAIL bi  n=%0d in_0=%b in_1=%0d : got %b exp %0d",
                             n, a, in1b, o_out_bi, out_b);
                    fails = fails + 1;
                end

                // clock edge advances the counters for the next cycle
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS mul_csg: %0d/%0d vectors (unipolar+bipolar)", n, n);
        else
            $display("FAIL mul_csg: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
