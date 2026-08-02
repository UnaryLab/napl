`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "sync_skewed/vec/sync_skewed_params.vh"
// Python golden outputs use the pre-update counter and are checked before each
// posedge. RST requests active-low reset before replay continues.
// Co-sim: make test OP=sync_skewed


module sync_skewed_tb;
    reg  clk;
    reg  rst_n;
    reg  in_1, in_2;
    wire out_1, out_2;

    sync_skewed #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input_1  (in_1),
        .i_input_2  (in_2),
        .o_out_1 (out_1),
        .o_out_2 (out_2)
    );

    integer fd, code, n, fails;
    reg a, b, exp_1, exp_2;
    reg [8*8-1:0] tag;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_1 = 1'b0;
        in_2 = 1'b0;

        // Release reset on a negedge so no update precedes the first check.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sync_skewed.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sync_skewed.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            // Rows are "in_1 in_2 exp_1 exp_2" or the reset sentinel "RST".
            code = $fscanf(fd, "%s", tag);
            if (code != 1) begin
            end else if (tag == "RST") begin
                rst_n = 1'b0;
                @(posedge clk);
                @(negedge clk);
                rst_n = 1'b1;
            end else begin
                a = (tag == "1");
                code = $fscanf(fd, "%b %b %b\n", b, exp_1, exp_2);
                in_1 = a;
                in_2 = b;
                #1;
                n = n + 1;
                if (out_1 !== exp_1) begin
                    $display("FAIL cyc=%0d in_1=%b in_2=%b : out_1 got %b exp %b", n, a, b, out_1, exp_1);
                    fails = fails + 1;
                end
                if (out_2 !== exp_2) begin
                    $display("FAIL cyc=%0d in_1=%b in_2=%b : out_2 got %b exp %b", n, a, b, out_2, exp_2);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sync_skewed: %0d/%0d vectors", n, n);
        else
            $display("FAIL sync_skewed: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
