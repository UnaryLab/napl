`timescale 1ns/1ps
`default_nettype none
// GEN_DEPTH is emitted by gen/gen_div_cordiv.py from the op config (= the test's
// div_cordiv_config), so the DUT parameter is inherited from the Python model.
// iverilog resolves this include relative to the compile cwd (hw/).
`include "div_cordiv/vec/div_cordiv_params.vh"
//==============================================================================
// Self-checking testbench for div_cordiv.
//
// Reads golden vectors produced by gen/gen_div_cordiv.py (from the napl Python
// model) and asserts the RTL reproduces them cycle by cycle. Prints "PASS ..."
// iff every vector matches; the Makefile greps for that line to decide the exit
// status.
//
// div_cordiv is stateful: o_quotient is combinational in the inputs and the
// current state, and the state advances on each posedge i_clk. Per vector we
// drive the inputs, let the combinational quotient settle, check it, then pulse
// the clock once to advance the state (one posedge == one Python forward()).
// i_rst_n is pulsed low first so the co-sim starts from the model's reset state.
//
// The vector file may also contain a lone "R" line: a MID-STREAM reset marker.
// On it the tb re-pulses i_rst_n low (matching model.reset()) so reset
// equivalence is proven from a dirtied operating state, not only at power-on.
//
// Run (from src/napl/hw/):
//   make test OP=div_cordiv
//==============================================================================
module div_cordiv_tb;
    reg  clk, rst_n;
    reg  dividend, divisor;
    wire quotient;

    div_cordiv #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_dividend(dividend),
        .i_divisor(divisor),
        .o_quotient(quotient)
    );

    integer fd, code, n, fails;
    reg [7:0] tag;
    reg dv, ds, exp_q;

    // Pulse i_rst_n low across one posedge to load the model's reset() state.
    task do_reset;
        begin
            rst_n = 1'b0;
            clk = 1'b1; #1;   // async reset fires here (buffer=0, idx=0)
            clk = 1'b0; #1;
            rst_n = 1'b1;
            #1;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b1;
        dividend = 1'b0;
        divisor = 1'b0;
        n = 0;
        fails = 0;

        // Power-on reset before the first stream.
        do_reset;

        fd = $fopen("vec/div_cordiv.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/div_cordiv.vec (run `make vectors` first)");
            $finish;
        end

        while (!$feof(fd)) begin
            // Peek the first token of the line: "R" is a mid-stream reset marker,
            // otherwise it is the dividend bit of a "dv ds q" vector line.
            code = $fscanf(fd, "%s", tag);
            if (code == 1) begin
                if (tag == "R") begin
                    // Mid-stream reset from a dirtied state: must re-converge.
                    do_reset;
                end else begin
                    dv = (tag == "1");
                    code = $fscanf(fd, "%b %b\n", ds, exp_q);
                    if (code == 2) begin
                        dividend = dv;
                        divisor  = ds;
                        #1;                  // let the combinational quotient settle
                        n = n + 1;
                        if (quotient !== exp_q) begin
                            $display("FAIL cyc=%0d dividend=%b divisor=%b : got %b exp %b",
                                     n, dv, ds, quotient, exp_q);
                            fails = fails + 1;
                        end
                        // Advance the state by one Python forward() timestep.
                        clk = 1'b1; #1;
                        clk = 1'b0; #1;
                    end
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS div_cordiv: %0d/%0d vectors", n, n);
        else
            $display("FAIL div_cordiv: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
