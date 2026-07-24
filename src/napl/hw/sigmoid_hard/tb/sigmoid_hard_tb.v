`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for sigmoid_hard.
//
// Replays golden vectors produced by gen/gen_sigmoid_hard.py (from the napl
// Python model) cycle by cycle and asserts the RTL reproduces them. Prints
// "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// sigmoid_hard is stateful (one posedge i_clk per Python forward() timestep).
// Each vector line is "<i_rst_n> <i_in> <o_out>". A line with i_rst_n=0 marks a
// cycle where the Python model was reset() (accumulator <- 0) *before*
// producing that cycle's output; the testbench holds its active-low reset low
// across that cycle so the accumulator is 0 when the combinational output is
// sampled. This replays the gen script's t=0 reset AND its mid-stream reset,
// proving the RTL's i_rst_n matches the model's reset from a dirtied state.
//
// Output is combinational in (acc, i_in), so it is sampled in the same cycle
// the input is applied.
//
// Run (from src/napl/hw/):
//   make test OP=sigmoid_hard
//==============================================================================
module sigmoid_hard_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_in;
    wire o_out;

    sigmoid_hard dut (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_in   (i_in),
        .o_out  (o_out)
    );

    // 10 ns clock
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_n_s, in_s, exp_out;

    initial begin
        fd = $fopen("vec/sigmoid_hard.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sigmoid_hard.vec (run `make vectors` first)");
            $finish;
        end

        // Settle on a falling edge before driving the first vector.
        i_in    = 1'b0;
        i_rst_n = 1'b1;
        @(negedge i_clk);

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst_n_s, in_s, exp_out);
            if (code == 3) begin
                // Drive reset and input just after a falling edge. On a reset
                // cycle (rst_n_s=0) the async active-low reset zeroes the
                // accumulator immediately, so the combinational output reflects
                // acc=0 with this cycle's input -- matching the model, which is
                // reset() before this cycle's forward().
                i_rst_n = rst_n_s;
                i_in    = in_s;
                #1;                 // let reset + combinational output settle
                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL cycle %0d: i_rst_n=%b i_in=%b got %b exp %b",
                             n, rst_n_s, in_s, o_out, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);   // commit the accumulator update
                i_rst_n = 1'b1;     // reset is a single-cycle pulse
                @(negedge i_clk);   // align to mid-cycle for the next sample
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sigmoid_hard: %0d/%0d vectors", n, n);
        else
            $display("FAIL sigmoid_hard: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
