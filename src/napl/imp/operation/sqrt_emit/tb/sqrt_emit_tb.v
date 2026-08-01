`timescale 1ns/1ps
`default_nettype none
// Python golden outputs for both polarities are checked before each posedge
// advances state. Reset clears state and loads the alternating shift register.
// Co-sim: make test OP=sqrt_emit
module sqrt_emit_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bip;

    sqrt_emit_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_uni)
    );

    sqrt_emit_bipolar dut_bip (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_bip)
    );

    integer fd, code, n, fails;
    reg a, rst, exp_u, exp_b;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Release reset on a negedge so vector 0 precedes the next state update.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_emit.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_emit.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", a, rst, exp_u, exp_b);
            if (code == 4) begin
                if (rst) begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                n = n + 1;
                in_bit = a;
                #1;
                if (out_uni !== exp_u) begin
                    $display("FAIL uni cyc=%0d in=%b : got %b exp %b", n, a, out_uni, exp_u);
                    fails = fails + 1;
                end
                if (out_bip !== exp_b) begin
                    $display("FAIL bip cyc=%0d in=%b : got %b exp %b", n, a, out_bip, exp_b);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_emit: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL sqrt_emit: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
