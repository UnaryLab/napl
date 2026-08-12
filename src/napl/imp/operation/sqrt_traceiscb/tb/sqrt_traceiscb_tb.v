`timescale 1ns/1ps
`default_nettype none
// Python golden outputs for both polarities are checked before each posedge
// advances state. The reset column requests active-low reset before its row.
// Co-sim: make test OP=sqrt_traceiscb


module sqrt_traceiscb_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bi;

    sqrt_traceiscb_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_output   (out_uni)
    );

    sqrt_traceiscb_bipolar dut_bi (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_output   (out_bi)
    );

    integer fd, code, n, fails;
    reg a, exp_uni, exp_bi, rst_mark;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Release reset on the negedge immediately before the first checked row.
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_traceiscb.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_traceiscb.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_mark, a, exp_uni, exp_bi);
            if (code == 4) begin
                if (rst_mark) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                n = n + 1;
                in_bit = a;
                #1;
                if (out_uni !== exp_uni) begin
                    $display("FAIL uni cyc=%0d in=%b : got %b exp %b", n, a, out_uni, exp_uni);
                    fails = fails + 1;
                end
                if (out_bi !== exp_bi) begin
                    $display("FAIL bi  cyc=%0d in=%b : got %b exp %b", n, a, out_bi, exp_bi);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_traceiscb: %0d/%0d vectors (uni+bi)", n, n);
        else
            $display("FAIL sqrt_traceiscb: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
