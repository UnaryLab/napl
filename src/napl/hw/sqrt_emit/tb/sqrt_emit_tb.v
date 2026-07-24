`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for sqrt_emit (unipolar + bipolar).
//
// Reads golden vectors produced by gen/gen_sqrt_emit.py (from the napl Python
// model) and asserts both polarity variants reproduce them cycle by cycle.
//
// Timing contract (matches the Python forward()): o_out at timestep t is
// combinational in i_in given the cycle-t registers, then the new state is
// clocked in. So per cycle we (1) drive i_in, let it settle combinationally,
// (2) check o_out == expected, then (3) pulse one posedge i_clk to commit the
// state update. i_rst_n is held low first so the co-sim starts from the exact
// post-reset() state (emit=0, acc=0, acc_b=0, sr[i]=i%2).
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/hw/):
//   make test OP=sqrt_emit
//==============================================================================
module sqrt_emit_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bip;

    sqrt_emit_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_in    (in_bit),
        .o_out   (out_uni)
    );

    sqrt_emit_bipolar dut_bip (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_in    (in_bit),
        .o_out   (out_bip)
    );

    integer fd, code, n, fails;
    reg a, rst, exp_u, exp_b;

    // Free-running clock: 10ns period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across a clock edge to load the exact post-
        // reset() state, then release it on the negedge that starts the first
        // checked cycle -- so the FIRST posedge commits timestep 0's state and
        // no extra posedge advances the registers before vector 0.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_emit.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_emit.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", a, rst, exp_u, exp_b);
            if (code == 4) begin
                // We are on a negedge: registers are stable. If this cycle is
                // flagged for a mid-stream reset, pulse active-low i_rst_n across
                // a posedge to reload the exact post-reset() state from a dirtied
                // state (proves reset equivalence), then release on a negedge so
                // this cycle is processed from the post-reset registers.
                if (rst) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                // Drive this cycle's input, let the combinational output settle,
                // and check it (the value forward() returns this timestep) before
                // the posedge commits the state update.
                n = n + 1;
                in_bit = a;
                #1;   // settle combinational paths
                if (out_uni !== exp_u) begin
                    $display("FAIL uni cyc=%0d in=%b : got %b exp %b", n, a, out_uni, exp_u);
                    fails = fails + 1;
                end
                if (out_bip !== exp_b) begin
                    $display("FAIL bip cyc=%0d in=%b : got %b exp %b", n, a, out_bip, exp_b);
                    fails = fails + 1;
                end
                @(posedge clk);   // commit state update for this timestep
                @(negedge clk);   // settle for the next check
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_emit: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL sqrt_emit: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
