`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for gt_rc.
//
// Reads golden vectors produced by gen/gen_gt_rc.py (from the napl Python
// model) and replays them cycle by cycle. gt_rc is stateful, so i_rst_n is
// pulsed low first to bring dff->1 and cnt->0 (the post-reset() model state),
// matching the generator which drives from model.reset() at t=0.
//
// Each vector line is "<rst_n> <in_0> <in_1> <out>". A line with rst_n==0 is a
// reset pulse (the generator's MID-STREAM model.reset()): the TB asserts
// i_rst_n low for that cycle (don't-care in_0/in_1/out), proving the RTL reset
// returns dff->1 / cnt->0 from a dirtied state. Otherwise it is a normal cycle.
//
// Timing: the model's output at timestep t is the result register read BEFORE
// its update at t. o_out tracks that register combinationally, so each cycle we
// drive (in_0, in_1), check o_out against the expected column, then pulse the
// clock to advance the state -- mirroring forward() returning the old dff then
// updating it.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/imp/):
//   make test OP=gt_rc
//==============================================================================
module gt_rc_tb;
    reg clk, rst_n, in_0, in_1;
    wire out;

    gt_rc dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input_0  (in_0),
        .i_input_1  (in_1),
        .o_out   (out)
    );

    integer fd, code, n, fails;
    reg rst_in, a, b, exp_out;

    initial begin
        clk   = 1'b0;
        in_0  = 1'b0;
        in_1  = 1'b0;

        // Pulse reset low -> dff=1, cnt=0 (post-reset() model state).
        rst_n = 1'b0;
        #1 rst_n = 1'b1;

        fd = $fopen("vec/gt_rc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/gt_rc.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_in, a, b, exp_out);
            if (code == 4) begin
                if (rst_in == 1'b0) begin
                    // Mid-stream reset pulse: assert i_rst_n low across a clock
                    // edge -> dff=1, cnt=0, mirroring model.reset(). No output
                    // check (don't-care row).
                    rst_n = 1'b0;
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                    rst_n = 1'b1; #1;
                end else begin
                    in_0 = a;
                    in_1 = b;
                    #1;                  // settle combinational o_out (current dff)
                    n = n + 1;
                    if (out !== exp_out) begin
                        $display("FAIL cycle %0d: in_0=%b in_1=%b : got %b exp %b",
                                 n - 1, a, b, out, exp_out);
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
            $display("PASS gt_rc: %0d/%0d vectors", n, n);
        else
            $display("FAIL gt_rc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
