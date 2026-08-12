`timescale 1ns/1ps
`default_nettype none
`include "eq_rc/vec/eq_rc_params.vh"
// Golden rows: <rst_n> <in0_u> <in1_u> <out_u> <in0_b> <in1_b> <out_b>.
// Two eq_rc instances (VALUE_GAIN 1 and 2) share the clock/reset and take the
// unipolar and bipolar streams. Output is combinational (pp_delay=0), so it is
// checked before the posedge that advances count/t. rst_n=0 rows pulse reset.
// Co-sim: make test OP=eq_rc


module eq_rc_tb;
    reg clk, rst_n;
    reg in0_u, in1_u, in0_b, in1_b;
    wire out_u, out_b;

    eq_rc #(.VALUE_GAIN(1), .TOL_NUM(`GEN_TOL_NUM), .TOL_DEN(`GEN_TOL_DEN), .TW(`GEN_TW)) dut_u (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input_0(in0_u),
        .i_input_1(in1_u),
        .o_output (out_u)
    );
    eq_rc #(.VALUE_GAIN(2), .TOL_NUM(`GEN_TOL_NUM), .TOL_DEN(`GEN_TOL_DEN), .TW(`GEN_TW)) dut_b (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input_0(in0_b),
        .i_input_1(in1_b),
        .o_output (out_b)
    );

    integer fd, code, n, fails;
    reg rst_in, a_u, b_u, e_u, a_b, b_b, e_b;

    initial begin
        clk = 1'b0;
        in0_u = 1'b0; in1_u = 1'b0; in0_b = 1'b0; in1_b = 1'b0;

        rst_n = 1'b0;
        #1 rst_n = 1'b1;

        fd = $fopen("vec/eq_rc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/eq_rc.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b %b %b %b\n",
                           rst_in, a_u, b_u, e_u, a_b, b_b, e_b);
            if (code == 7) begin
                if (rst_in == 1'b0) begin
                    // Shared mid-stream reset pulse: assert i_rst_n low across a
                    // clock edge -> count=0, t=0 in both instances. No output check.
                    rst_n = 1'b0;
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                    rst_n = 1'b1; #1;
                end else begin
                    in0_u = a_u; in1_u = b_u;
                    in0_b = a_b; in1_b = b_b;
                    #1;                  // settle combinational o_output (post-update)
                    n = n + 1;
                    if (out_u !== e_u) begin
                        $display("FAIL cycle %0d (uni): in0=%b in1=%b got %b exp %b",
                                 n - 1, a_u, b_u, out_u, e_u);
                        fails = fails + 1;
                    end
                    if (out_b !== e_b) begin
                        $display("FAIL cycle %0d (bip): in0=%b in1=%b got %b exp %b",
                                 n - 1, a_b, b_b, out_b, e_b);
                        fails = fails + 1;
                    end
                    // advance state: one posedge consumes this cycle's inputs.
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS eq_rc: %0d/%0d vectors", n, n);
        else
            $display("FAIL eq_rc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
