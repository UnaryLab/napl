`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for relu_sat.
//
// Reads golden vectors produced by gen/gen_relu_sat.py (from the napl Python
// model) and asserts the RTL reproduces them cycle-for-cycle. relu_sat is
// stateful and combinational from state (pp_delay=0): each vector row is one
// timestep "<rst> <i_in> <o_out>".
//
//   * rst=0 : a normal timestep. Drive i_in, sample o_out in the same cycle
//             (combinational from the accumulators), then pulse one posedge
//             i_clk to commit the accumulator update.
//   * rst=1 : the model called reset() BEFORE this timestep. Assert i_rst_n low
//             across a posedge so the async reset loads acc_sub=acc_add=0, then
//             release it and proceed as a normal cycle. This appears at the
//             stream start AND mid-stream (after the accumulators are dirtied),
//             proving the RTL's async i_rst_n reproduces model.reset() exactly.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/implementation/):
//   make test OP=relu_sat
//==============================================================================
module relu_sat_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_in;
    wire o_out;

    relu_sat dut (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_in   (i_in),
        .o_out  (o_out)
    );

    // 10ns clock
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

        i_in    = 1'b0;
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
                // Drive the input on a negedge so it is stable; o_out is
                // combinational from the (now possibly reset) accumulators.
                i_in = in_b;
                #1;                     // let the combinational output settle
                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL cycle %0d: rst=%b i_in=%b got %b exp %b",
                             n, rst_b, in_b, o_out, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the accumulator state
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
