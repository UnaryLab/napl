`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH mirrors the Python model configuration.
`include "shiftreg/vec/shiftreg_params.vh"
// Python golden output is the oldest cell before the posedge shifts input.
// R requests active-low reset to the alternating pattern.
// Co-sim: make test OP=shiftreg
module shiftreg_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_bit;

    shiftreg #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_out   (out_bit)
    );

    integer fd, code, n, fails;
    reg [8*8-1:0] tok;
    reg a, exp_out;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across a clock edge to load reg[i]=i%2 (the
        // async reset fires on the posedge while rst_n is low), then release it
        // on a negedge. No posedge may occur between release and the first check,
        // or the delay line would shift once and the opening reset pattern would
        // be off by one.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);   // async reset loads reg[i]=i%2 here
        @(negedge clk);
        rst_n = 1'b1;     // released; next posedge (inside the loop) is timestep 1

        fd = $fopen("vec/shiftreg.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/shiftreg.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != `GEN_DEPTH) begin
            $display(
                "FAIL shiftreg: observed latency %0d, expected pp_delay %0d",
                `GEN_DEPTH,
                `GEN_PP_DELAY
            );
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s", tok);
            if (code != 1) begin
                code = 0;
            end else if (tok == "R") begin
                rst_n  = 1'b0;
                in_bit = 1'b0;
                @(posedge clk);
                @(negedge clk);
                rst_n = 1'b1;
            end else begin
                a = (tok[7:0] == "1");
                code = $fscanf(fd, "%b\n", exp_out);
                n = n + 1;
                if (out_bit !== exp_out) begin
                    $display("FAIL cyc=%0d in=%b : got %b exp %b", n, a, out_bit, exp_out);
                    fails = fails + 1;
                end
                in_bit = a;
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS shiftreg: %0d/%0d vectors", n, n);
        else
            $display("FAIL shiftreg: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
