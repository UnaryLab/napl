`timescale 1ns/1ps
`default_nettype none
// GEN_WIDTH is emitted by gen/gen_sync_skewed.py from the op config (= the test's
// sync_skewed_config), so the DUT parameter is inherited from the Python model.
// iverilog resolves this include relative to the compile cwd (implementation/).
`include "sync_skewed/vec/sync_skewed_params.vh"
//==============================================================================
// Self-checking testbench for sync_skewed.
//
// Reads golden vectors produced by gen/gen_sync_skewed.py (from the napl Python
// model) and asserts the skewed synchronizer reproduces them cycle by cycle.
//
// Timing contract (matches the Python forward()): at timestep t the outputs are
// combinational in (i_in_1, i_in_2, cnt) where cnt is the value BEFORE this
// timestep's update. So per cycle we (1) drive this cycle's inputs, (2) let the
// combinational outputs settle and check them against the current register
// state, then (3) pulse one posedge i_clk to perform the counter update.
// i_rst_n is held low first so the co-sim starts from the exact post-reset()
// state (cnt = 0).
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/implementation/):
//   make test OP=sync_skewed
//==============================================================================
module sync_skewed_tb;
    reg  clk;
    reg  rst_n;
    reg  in_1, in_2;
    wire out_1, out_2;

    sync_skewed #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_in_1  (in_1),
        .i_in_2  (in_2),
        .o_out_1 (out_1),
        .o_out_2 (out_2)
    );

    integer fd, code, n, fails;
    reg a, b, exp_1, exp_2;
    reg [8*8-1:0] tag;

    // Free-running clock: 10ns period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_1 = 1'b0;
        in_2 = 1'b0;

        // Assert active-low reset across a clock edge to load cnt = 0 (the async
        // reset fires on the posedge while rst_n is low), then release it on a
        // negedge so no spurious posedge updates cnt before the first check.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);   // async reset loads cnt = 0 here
        @(negedge clk);
        rst_n = 1'b1;     // released; next posedge (inside the loop) is the update

        fd = $fopen("vec/sync_skewed.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sync_skewed.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            // Each line is either four spike bits or the literal "RST" marking a
            // mid-stream reset (emitted by gen after model.reset()). Peek the
            // first whitespace token: if it is "RST" we re-assert i_rst_n to
            // reproduce the model's reset() from a dirtied counter; otherwise the
            // line carries (in_1 in_2 exp_1 exp_2).
            code = $fscanf(fd, "%s", tag);
            if (code != 1) begin
                // no more tokens (trailing newline / EOF)
            end else if (tag == "RST") begin
                // Async reset across a posedge to clear cnt = 0, released on a
                // negedge so no spurious posedge updates cnt before the next check.
                rst_n = 1'b0;
                @(posedge clk);
                @(negedge clk);
                rst_n = 1'b1;
            end else begin
                // tag holds the in_1 bit; read the remaining three on this line.
                a = (tag == "1");
                code = $fscanf(fd, "%b %b %b\n", b, exp_1, exp_2);
                // On a negedge: registers are stable. Drive this cycle's inputs,
                // let the combinational outputs settle, then check before the
                // posedge updates cnt.
                in_1 = a;
                in_2 = b;
                #1;
                n = n + 1;
                if (out_1 !== exp_1) begin
                    $display("FAIL cyc=%0d in_1=%b in_2=%b : out_1 got %b exp %b", n, a, b, out_1, exp_1);
                    fails = fails + 1;
                end
                if (out_2 !== exp_2) begin
                    $display("FAIL cyc=%0d in_1=%b in_2=%b : out_2 got %b exp %b", n, a, b, out_2, exp_2);
                    fails = fails + 1;
                end
                @(posedge clk);   // update cnt
                @(negedge clk);   // settle for the next check
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sync_skewed: %0d/%0d vectors", n, n);
        else
            $display("FAIL sync_skewed: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
