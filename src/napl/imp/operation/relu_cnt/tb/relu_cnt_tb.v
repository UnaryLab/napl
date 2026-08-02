`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "relu_cnt/vec/relu_cnt_params.vh"
// Python golden output is checked before each posedge updates acc. The reset
// column requests active-low reset to HALF before its row.
// Co-sim: make test OP=relu_cnt


module relu_cnt_tb;
    localparam WIDTH = `GEN_WIDTH;

    reg  clk, rst_n, in;
    wire out;

    relu_cnt #(.WIDTH(WIDTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in),
        .o_out   (out)
    );

    integer fd, code, n, fails;
    reg a, exp_out, rst;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in    = 1'b0;
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/relu_cnt.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/relu_cnt.vec (run `make vectors` first)");
            $finish;
        end

        // Reset to the model's post-reset() state (acc = HALF). Hold rst_n low
        // across a posedge, then deassert on the falling edge so the first
        // driven vector sees acc = HALF (no stray advancing edge in between).
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", a, exp_out, rst);
            if (code == 3) begin
                // Mid-stream reset: pulse i_rst_n low across a posedge so the
                // counter reloads acc = HALF, matching the model's reset().
                if (rst) begin
                    @(negedge clk);
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                in = a;
                #1;
                n = n + 1;
                if (out !== exp_out) begin
                    $display("FAIL t=%0d in=%b : got %b exp %b", n, a, out, exp_out);
                    fails = fails + 1;
                end
                @(posedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS relu_cnt: %0d/%0d vectors", n, n);
        else
            $display("FAIL relu_cnt: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
