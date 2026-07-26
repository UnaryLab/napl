`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for min_rc.
//
// Reads golden vectors produced by gen/gen_min_rc.py (from the napl Python
// model) and drives the same per-cycle spike stream into the DUT. Pulses
// i_rst_n low first so the co-sim starts from the post-reset() state (cnt=0,
// dff=0). Each vector is applied for exactly one posedge i_clk -- one Python
// forward() timestep == one clock -- and the registered/combinational outputs
// are sampled just before the edge. Prints "PASS ..." iff every vector matches;
// the Makefile greps for that line to decide the exit status.
//
// Run (from src/napl/imp/):
//   make test OP=min_rc
//==============================================================================
module min_rc_tb;
    reg  clk, rst_n;
    reg  in_0, in_1;
    wire o_min, o_argmin;

    min_rc dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input_0  (in_0),
        .i_input_1  (in_1),
        .o_min   (o_min),
        .o_argmin(o_argmin)
    );

    // 10ns clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer fd, code, n, fails;
    reg a, b, exp_min, exp_arg, rst_mid;

    initial begin
        fd = $fopen("vec/min_rc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/min_rc.vec (run `make vectors` first)");
            $finish;
        end

        // assert reset (active low) -> post-reset() state
        in_0  = 1'b0;
        in_1  = 1'b0;
        rst_n = 1'b0;
        @(posedge clk);
        #1 rst_n = 1'b1;   // release reset shortly after the edge

        n     = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b %b\n", a, b, exp_min, exp_arg, rst_mid);
            if (code == 5) begin
                // mid-stream reset: pulse i_rst_n low so the DUT re-enters the
                // post-reset() state (cnt=0, dff=0) exactly where the Python
                // model calls reset() again, proving reset equivalence from a
                // dirtied state.
                if (rst_mid) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    #1 rst_n = 1'b1;
                end
                // drive inputs for this timestep, settle combinational paths
                in_0 = a;
                in_1 = b;
                #1;
                n = n + 1;
                if (o_min !== exp_min) begin
                    $display("FAIL[min] cyc=%0d in_0=%b in_1=%b : got %b exp %b",
                             n, a, b, o_min, exp_min);
                    fails = fails + 1;
                end
                if (o_argmin !== exp_arg) begin
                    $display("FAIL[arg] cyc=%0d in_0=%b in_1=%b : got %b exp %b",
                             n, a, b, o_argmin, exp_arg);
                    fails = fails + 1;
                end
                // advance one timestep: latch dff/cnt next-state
                @(posedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS min_rc: %0d/%0d vectors", n, n);
        else
            $display("FAIL min_rc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
