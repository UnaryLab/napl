`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for sqrt_tracejkff (unipolar + bipolar variants).
//
// Reads golden vectors produced by gen/gen_sqrt_tracejkff.py (from the napl
// Python model) and asserts both polarity variants reproduce them cycle by
// cycle.
//
// Timing contract (matches the Python forward()): o_out at timestep t is
// combinational in i_in given the cycle-t trace/acc registers, then the new
// state is clocked in. So per cycle we (1) drive i_in, let it settle, (2) check
// o_out against the expected column BEFORE the committing posedge, then (3)
// pulse one posedge i_clk to commit the state update. Checking before the
// posedge (rather than on the next negedge) is required: otherwise the posedge
// that follows reset-deassert advances the trace register one cycle too early,
// an off-by-one that only shows up once cycle 0's input drives trace to 1.
// i_rst_n is held low first so the co-sim starts from the exact post-reset()
// state (trace=0, acc=0).
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/hw/):
//   make test OP=sqrt_tracejkff
//==============================================================================
module sqrt_tracejkff_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bip;

    sqrt_tracejkff_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_in    (in_bit),
        .o_out   (out_uni)
    );

    sqrt_tracejkff_bipolar dut_bip (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_in    (in_bit),
        .o_out   (out_bip)
    );

    integer fd, code, n, fails;
    reg a, exp_u, exp_b;
    reg [8*8-1:0] tok;   // first whitespace-delimited token of a vec line

    // Free-running clock: 10ns period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across clock edges to load trace=0, acc=0.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_tracejkff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_tracejkff.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            // Read the first token of the line. "R" is a mid-stream reset
            // sentinel; otherwise it is this cycle's input bit and the line
            // also carries the two expected output columns.
            code = $fscanf(fd, "%s", tok);
            if (code == 1) begin
                if (tok == "R") begin
                    // Mid-stream reset: pulse active-low i_rst_n across an edge
                    // to clear trace=0, acc=0, mirroring the model's reset from
                    // a dirtied state. We are on a negedge here.
                    rst_n = 1'b0;
                    @(posedge clk);   // commit the reset into the registers
                    @(negedge clk);
                    rst_n = 1'b1;
                end else begin
                    // tok holds the input bit; read the two expected columns.
                    a     = (tok == "1");
                    code  = $fscanf(fd, "%b %b\n", exp_u, exp_b);
                    // We are on a negedge: registers are stable. Drive this
                    // cycle's input, let the combinational output settle, and
                    // check it (the value forward() returns this timestep)
                    // BEFORE the posedge commits the state update.
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
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_tracejkff: %0d/%0d vectors", n, n);
        else
            $display("FAIL sqrt_tracejkff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
