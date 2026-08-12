`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH mirrors the Python model configuration.
`include "add_desync/vec/add_desync_params.vh"
// Python golden outputs use the pre-update counter and are checked before each
// posedge. RST requests active-low reset before replay continues.
// Co-sim: make test OP=add_desync


module add_desync_tb;
    reg  clk;
    reg  rst_n;
    reg  in_0, in_1;
    wire out;

    add_desync #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk     (clk),
        .i_rst_n   (rst_n),
        .i_input_0 (in_0),
        .i_input_1 (in_1),
        .o_output  (out)
    );

    integer fd, code, n, fails;
    reg a, b, exp_out;
    reg [8*8-1:0] tag;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_0 = 1'b0;
        in_1 = 1'b0;

        // Release reset on a negedge so no update precedes the first check.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/add_desync.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/add_desync.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // The output is combinational in the pre-update counter.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL add_desync: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            // Rows are "in_0 in_1 exp_out" or the reset sentinel "RST".
            code = $fscanf(fd, "%s", tag);
            if (code != 1) begin
                code = 0;
            end else if (tag == "RST") begin
                rst_n = 1'b0;
                @(posedge clk);
                @(negedge clk);
                rst_n = 1'b1;
            end else begin
                a = (tag == "1");
                code = $fscanf(fd, "%b %b\n", b, exp_out);
                in_0 = a;
                in_1 = b;
                #1;
                n = n + 1;
                if (out !== exp_out) begin
                    $display("FAIL cyc=%0d in_0=%b in_1=%b : out got %b exp %b", n, a, b, out, exp_out);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        // A vec file that lost or gained rows fails instead of passing on the rest.
        if (n != `GEN_VECTORS) begin
            $display("FAIL add_desync: consumed %0d rows, generator emitted %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS add_desync: %0d/%0d vectors", n, n);
        else
            $display("FAIL add_desync: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
