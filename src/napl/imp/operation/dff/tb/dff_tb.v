`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH mirrors the Python model configuration.
`include "dff/vec/dff_params.vh"
// The Python golden output is the oldest cell before the posedge shifts input.
// R requests an active-low reset that clears every cell.
// Co-sim: make test OP=dff


module dff_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_bit;

    dff #(.DEPTH(`GEN_DEPTH)) dut (
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

        // Assert active-low reset across a clock edge to load the all-zeros
        // FIFO (the async reset fires on the posedge while rst_n is low), then
        // release it on a negedge. No posedge may occur between release and the
        // first check, or the delay line would shift once.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);   // async reset loads all zeros here
        @(negedge clk);
        rst_n = 1'b1;     // released; next posedge (inside the loop) is timestep 1

        fd = $fopen("vec/dff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/dff.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != `GEN_DEPTH) begin
            $display(
                "FAIL dff: observed latency %0d, expected pp_delay %0d",
                `GEN_DEPTH,
                `GEN_PP_DELAY
            );
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            // Try to read a data row "<in> <out>". A lone "R" line is a reset
            // marker. We read the first whitespace-delimited token as a string
            // and branch on it.
            code = $fscanf(fd, "%s", tok);
            if (code != 1) begin
                // EOF or blank: stop.
                code = 0;
            end else if (tok == "R") begin
                // Mid-stream reset: pulse i_rst_n low across a posedge to reload
                // the all-zeros state, mirroring model.reset().
                rst_n  = 1'b0;
                in_bit = 1'b0;
                @(posedge clk);   // async reset reloads all zeros
                @(negedge clk);
                rst_n  = 1'b1;    // released; back on a negedge, registers stable
            end else begin
                a       = (tok[7:0] == "1");
                code    = $fscanf(fd, "%b\n", exp_out);
                n = n + 1;
                if (out_bit !== exp_out) begin
                    $display("FAIL cyc=%0d in=%b : got %b exp %b", n, a, out_bit, exp_out);
                    fails = fails + 1;
                end
                in_bit = a;
                @(posedge clk);   // shift: emit oldest, append in_bit
                @(negedge clk);   // settle for the next check
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS dff: %0d/%0d vectors", n, n);
        else
            $display("FAIL dff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
