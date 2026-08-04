`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH/FRAC mirror the Python model configuration.
`include "encode/vec/encode_params.vh"
// Python golden rows are <rst> <i_input> <o_spike>; the spike is checked in the
// arrival cycle, before the posedge advances the sequence index. rst=1 requests an
// active-low reset to a zero index before its row.
// Co-sim: make test OP=encode


module encode_tb;
    localparam WIDTH = `GEN_WIDTH;
    localparam FRAC  = `GEN_FRAC;

    reg  clk, rst_n;
    reg  [FRAC:0] value;
    wire spike;

    encode #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input (value),
        .o_spike (spike)
    );

    integer fd, code, n, fails;
    integer in_value, exp_spike;
    reg rst;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        value = {(FRAC+1){1'b0}};
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/encode.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/encode.vec (run `make vectors` first)");
            $finish;
        end

        // pp_delay is 0: the spike is combinational in the arrival cycle.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL encode: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // Reset to the model's post-reset() state (index = 0). Hold rst_n low
        // across a posedge, then deassert on the falling edge so the first driven
        // vector sees index 0 with no stray advancing edge in between.
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %d %b\n", rst, in_value, exp_spike);
            if (code == 3) begin
                if (rst) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                // Drive from the negedge: the index register samples on the
                // posedge, so an input changing at that edge would race it.
                value = in_value[FRAC:0];
                #1;
                n = n + 1;
                if (spike !== exp_spike[0]) begin
                    $display("FAIL t=%0d value=%0d : got %b exp %0d", n, in_value, spike, exp_spike);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS encode: %0d/%0d vectors (unipolar+bipolar probabilities)", n, n);
        else
            $display("FAIL encode: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
