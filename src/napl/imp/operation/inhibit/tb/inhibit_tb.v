`timescale 1ns/1ps
`default_nettype none
// Python golden rows are <rst_n> <in_data> <in_inhibit> <out>. The DUT is clockless:
// both inputs are consumed combinationally and the level-sensitive latch updates
// with them, so the output is checked in the arrival cycle; rst_n=0 restores
// latch=0. The clock here only paces the vector stream.
// Co-sim: make test OP=inhibit


module inhibit_tb;
    reg clk, rst_n, in_data, in_inhibit;
    wire out;

    inhibit dut (
        .i_clk           (clk),
        .i_rst_n         (rst_n),
        .i_input_data    (in_data),
        .i_input_inhibit (in_inhibit),
        .o_output        (out)
    );

    integer fd, code, n, fails;
    reg rst_in, a, b, exp_out;

    initial begin
        clk  = 1'b0;
        in_data = 1'b0;
        in_inhibit = 1'b0;

        rst_n = 1'b0;
        #1 rst_n = 1'b1;

        fd = $fopen("vec/inhibit.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/inhibit.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_in, a, b, exp_out);
            if (code == 4) begin
                if (rst_in == 1'b0) begin
                    // Mid-stream reset pulse: assert i_rst_n low -> latch=0,
                    // mirroring model.reset(). No output check (don't-care row).
                    rst_n = 1'b0;
                    in_data = 1'b0;
                    in_inhibit = 1'b0;
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                    rst_n = 1'b1; #1;
                end else begin
                    {in_data, in_inhibit} = {a, b}; // inputs of one timestep update atomically so the level latch never sees a half-updated vector
                    #1;                  // settle the combinational o_output
                    n = n + 1;
                    if (out !== exp_out) begin
                        $display("FAIL cycle %0d: in_data=%b in_inhibit=%b : got %b exp %b",
                                 n - 1, a, b, out, exp_out);
                        fails = fails + 1;
                    end
                    // pace the vector stream; the DUT ignores this clock.
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS inhibit: %0d/%0d vectors", n, n);
        else
            $display("FAIL inhibit: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
