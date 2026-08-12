`timescale 1ns/1ps
`default_nettype none
// Python golden rows are <rst_n> <input> <out>. Output is checked before each
// posedge updates acc; rst_n=0 first clears acc.
// Co-sim: make test OP=sigmoid_hard


module sigmoid_hard_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output;

    sigmoid_hard dut (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input),
        .o_output  (o_output)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_n_s, in_s, exp_out;

    initial begin
        fd = $fopen("vec/sigmoid_hard.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sigmoid_hard.vec (run `make vectors` first)");
            $finish;
        end

        i_input    = 1'b0;
        i_rst_n = 1'b1;
        @(negedge i_clk);

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst_n_s, in_s, exp_out);
            if (code == 3) begin
                // Reset is applied before this row's combinational output.
                i_rst_n = rst_n_s;
                i_input    = in_s;
                #1;
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL cycle %0d: i_rst_n=%b i_input=%b got %b exp %b",
                             n, rst_n_s, in_s, o_output, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);
                i_rst_n = 1'b1;
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sigmoid_hard: %0d/%0d vectors", n, n);
        else
            $display("FAIL sigmoid_hard: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
