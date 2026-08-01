`timescale 1ns/1ps
`default_nettype none
// Python golden rows are <rst> <input> <out>. Output is checked before each
// posedge updates state; rst=1 first clears both accumulators.
// Co-sim: make test OP=relu_sat
module relu_sat_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_out;

    relu_sat dut (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input),
        .o_out  (o_out)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_b, in_b, exp_out;

    initial begin
        fd = $fopen("vec/relu_sat.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/relu_sat.vec (run `make vectors` first)");
            $finish;
        end

        i_input    = 1'b0;
        i_rst_n = 1'b1;
        @(negedge i_clk);

        n     = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst_b, in_b, exp_out);
            if (code == 3) begin
                if (rst_b) begin
                    // Async reset to the model's post-reset() state (acc=0):
                    // hold i_rst_n low across a posedge, then release on a
                    // negedge so the next sample sees the cleared accumulators.
                    i_rst_n = 1'b0;
                    @(posedge i_clk);
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                i_input = in_b;
                #1;
                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL cycle %0d: rst=%b i_input=%b got %b exp %b",
                             n, rst_b, in_b, o_out, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS relu_sat: %0d/%0d vectors", n, n);
        else
            $display("FAIL relu_sat: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
