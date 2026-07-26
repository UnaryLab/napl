`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for sqrt_traceiscb (unipolar + bipolar variants).
//
// Reads golden vectors produced by gen/gen_sqrt_traceiscb.py (from the napl
// Python model) and asserts both polarity DUTs reproduce them cycle by cycle.
//
// Timing contract (matches the Python forward()): the output at timestep t is
// combinational from the current input and the registered state at the START of
// the cycle. So per cycle we (1) drive i_input, (2) let the combinational o_out
// settle, (3) check it against the expected column, then (4) pulse one posedge
// i_clk to advance the registered state. i_rst_n is held low first so the co-sim
// starts from the exact post-reset() state (all registers 0).
//
// Each vector also carries a reset marker (first column). When it is 1, the
// model applied reset() at the START of that cycle; the tb mirrors this by
// pulsing i_rst_n low across a posedge (reloading the post-reset() state) before
// driving the input. The generator injects one such mid-stream reset from a
// dirtied state, so the co-sim proves the RTL reset matches reset() at an
// arbitrary point, not only at t=0.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/imp/):
//   make test OP=sqrt_traceiscb
//==============================================================================
module sqrt_traceiscb_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bi;

    sqrt_traceiscb_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_uni)
    );

    sqrt_traceiscb_bipolar dut_bi (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_bi)
    );

    integer fd, code, n, fails;
    reg a, exp_uni, exp_bi, rst_mark;

    // Free-running clock: 10ns period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across a posedge so every register loads its
        // post-reset() value (all zero). Release reset ON the negedge that
        // immediately precedes the first checked cycle, so NO extra posedge
        // advances the state before the first input is driven (the first checked
        // state is the exact post-reset() state).
        rst_n = 1'b0;
        @(posedge clk);   // async reset already holds; this edge keeps state at 0
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_traceiscb.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_traceiscb.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_mark, a, exp_uni, exp_bi);
            if (code == 4) begin
                // We are on a negedge: registers are stable. If this cycle is a
                // reset cycle, pulse i_rst_n low across a posedge to reload the
                // post-reset() state (mirrors the model's mid-stream reset()),
                // then return to a negedge before driving the input.
                if (rst_mark) begin
                    rst_n = 1'b0;
                    @(posedge clk);   // async reset reloads all registers to 0
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                // Drive this cycle's input, let o_out (combinational) settle,
                // then check both DUTs.
                n = n + 1;
                in_bit = a;
                #1;  // let combinational outputs settle
                if (out_uni !== exp_uni) begin
                    $display("FAIL uni cyc=%0d in=%b : got %b exp %b", n, a, out_uni, exp_uni);
                    fails = fails + 1;
                end
                if (out_bi !== exp_bi) begin
                    $display("FAIL bi  cyc=%0d in=%b : got %b exp %b", n, a, out_bi, exp_bi);
                    fails = fails + 1;
                end
                @(posedge clk);   // advance registered state
                @(negedge clk);   // settle for the next check
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_traceiscb: %0d/%0d vectors (uni+bi)", n, n);
        else
            $display("FAIL sqrt_traceiscb: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
