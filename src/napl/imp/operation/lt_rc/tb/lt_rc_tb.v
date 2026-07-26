`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for lt_rc.
//
// Replays golden vectors produced by gen/gen_lt_rc.py (from the napl Python
// model). Each line is:  <rst> <in_0> <in_1> <out>.
//
// lt_rc is stateful and its output is the *registered* dff value (the state
// before this cycle's update), so for each vector:
//   1. if rst==1 this is the first cycle of a new reset segment -> pulse
//      i_rst_n low so cnt<=0, dff<=0 (the post-reset() state),
//   2. drive i_input_0/i_input_1 for the timestep,
//   3. o_out already holds the pre-edge dff (== model output), so compare it,
//   4. pulse i_clk to advance the registered state to the next timestep.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/imp/):
//   make test OP=lt_rc
//==============================================================================
module lt_rc_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_0, i_input_1;
    wire o_out;

    lt_rc dut (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input_0 (i_input_0),
        .i_input_1 (i_input_1),
        .o_out  (o_out)
    );

    integer fd, code, n, fails;
    reg rst, a, b, exp_out;

    // pulse the active-low reset: bring cnt/dff to the post-reset() state.
    task do_reset;
        begin
            i_rst_n = 1'b0;
            #1 i_clk = 1'b1;   // edge while reset asserted -> load reset state
            #1 i_clk = 1'b0;
            i_rst_n = 1'b1;
            #1;
        end
    endtask

    initial begin
        fd = $fopen("vec/lt_rc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/lt_rc.vec (run `make vectors` first)");
            $finish;
        end

        n     = 0;
        fails = 0;
        i_clk = 1'b0;
        i_rst_n = 1'b1;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst, a, b, exp_out);
            if (code == 4) begin
                if (rst) do_reset;
                i_input_0 = a;
                i_input_1 = b;
                #1;                       // let combinational o_out settle
                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL t=%0d rst=%b in_0=%b in_1=%b : got %b exp %b",
                             n, rst, a, b, o_out, exp_out);
                    fails = fails + 1;
                end
                // advance the registered state to the next timestep.
                #1 i_clk = 1'b1;
                #1 i_clk = 1'b0;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS lt_rc: %0d/%0d vectors", n, n);
        else
            $display("FAIL lt_rc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
