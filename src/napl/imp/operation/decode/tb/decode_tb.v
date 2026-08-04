`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "decode/vec/decode_params.vh"
// Python golden count is checked in the arrival cycle, before the posedge stores
// it. The reset column requests active-low reset to a zero count before its row.
// Co-sim: make test OP=decode


module decode_tb;
    localparam WIDTH = `GEN_WIDTH;

    reg  clk, rst_n, spike;
    wire [WIDTH:0] spike_count;

    decode #(.WIDTH(WIDTH)) dut (
        .i_clk         (clk),
        .i_rst_n       (rst_n),
        .i_spike       (spike),
        .o_spike_count (spike_count)
    );

    integer fd, code, n, fails;
    reg a, rst;
    integer exp_count;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        spike = 1'b0;
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/decode.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/decode.vec (run `make vectors` first)");
            $finish;
        end

        // pp_delay is 0: the count is combinational in the arrival cycle.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL decode: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // Reset to the model's post-reset() state (count = 0). Hold rst_n low
        // across a posedge, then deassert on the falling edge so the first
        // driven vector sees count = 0 (no stray advancing edge in between).
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %d %b\n", a, exp_count, rst);
            if (code == 3) begin
                if (rst) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                // Drive from the negedge: the count register samples i_spike on
                // the posedge, so an input changing at that edge would race it.
                spike = a;
                #1;
                n = n + 1;
                if (spike_count !== exp_count[WIDTH:0]) begin
                    $display("FAIL t=%0d spike=%b : got %0d exp %0d", n, a, spike_count, exp_count);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS decode: %0d/%0d vectors", n, n);
        else
            $display("FAIL decode: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
