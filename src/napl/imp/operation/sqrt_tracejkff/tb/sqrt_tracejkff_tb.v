`timescale 1ns/1ps
`default_nettype none
// Python golden outputs must be checked before the committing posedge; checking
// later advances trace one cycle early after reset. R clears trace and acc.
// Co-sim: make test OP=sqrt_tracejkff
module sqrt_tracejkff_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bip;

    sqrt_tracejkff_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_uni)
    );

    sqrt_tracejkff_bipolar dut_bip (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_bip)
    );

    integer fd, code, n, fails;
    reg a, exp_u, exp_b;
    reg [8*8-1:0] tok;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Reset loads trace=0 and acc=0.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_tracejkff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_tracejkff.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s", tok);
            if (code == 1) begin
                if (tok == "R") begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end else begin
                    a     = (tok == "1");
                    code  = $fscanf(fd, "%b %b\n", exp_u, exp_b);
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
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_tracejkff: %0d/%0d vectors", n, n);
        else
            $display("FAIL sqrt_tracejkff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
