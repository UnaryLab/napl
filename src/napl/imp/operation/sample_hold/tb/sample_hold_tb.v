`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH/TRIGGER mirror the Python model configuration.
`include "sample_hold/vec/sample_hold_params.vh"
// Python golden rows are <rst> <i_input> <o_output>; the output is checked in the
// arrival cycle, before the posedge advances state. rst=1 requests an active-low
// reset to the model's post-reset() state before its row.
// Co-sim: make test OP=sample_hold


module sample_hold_tb;
    localparam WIDTH   = `GEN_WIDTH;
    localparam TRIGGER = `GEN_TRIGGER;

    reg  clk, rst_n, spike;
    wire out;

    sample_hold #(.WIDTH(WIDTH), .TRIGGER(TRIGGER)) dut (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input  (spike),
        .o_output (out)
    );

    integer fd, code, n, fails;
    reg a, rst, exp_out;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        spike = 1'b0;
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/sample_hold.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sample_hold.vec (run `make vectors` first)");
            $finish;
        end

        // pp_delay is 0: the output is combinational in the arrival cycle.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL sample_hold: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // Reset to the model's post-reset() state. Hold rst_n low across a posedge,
        // then deassert on the falling edge so the first driven vector starts at t=0
        // with no stray advancing edge in between.
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst, a, exp_out);
            if (code == 3) begin
                if (rst) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                // Drive from the negedge: the state registers sample on the posedge,
                // so an input changing at that edge would race them.
                spike = a;
                #1;
                n = n + 1;
                if (out !== exp_out) begin
                    $display("FAIL t=%0d spike=%b : got %b exp %b", n, a, out, exp_out);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sample_hold: %0d/%0d vectors (unipolar+bipolar)", n, n);
        else
            $display("FAIL sample_hold: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
