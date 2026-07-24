`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for div_iscb (unipolar + bipolar).
//
// Reads golden vectors produced by gen/gen_div_iscb.py (from the napl Python
// model) and asserts both polarity variants reproduce them cycle-for-cycle.
//
// Timing contract: one Python forward() timestep == one posedge i_clk, and the
// quotient is combinational from the current inputs + current register state
// (pp_delay = 0). So for each vector we drive the inputs, let the combinational
// outputs settle, compare, THEN pulse the clock to advance the state. i_rst_n
// is pulsed low first so the co-sim starts from the model's post-reset() state.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/hw/):
//   make test OP=div_iscb
//==============================================================================
module div_iscb_tb;
    reg  clk;
    reg  rst_n;
    reg  dividend, divisor;

    wire q_uni, q_bi;

    div_iscb_unipolar dut_uni (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_dividend(dividend), .i_divisor(divisor),
        .o_quotient(q_uni)
    );
    div_iscb_bipolar dut_bi (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_dividend(dividend), .i_divisor(divisor),
        .o_quotient(q_bi)
    );

    integer fd, code, n, fails;
    reg rst_row;
    reg dd, ds, exp_uni, exp_bi;

    initial begin
        clk = 1'b0;
        rst_n = 1'b1;
        dividend = 1'b0;
        divisor = 1'b0;

        // Pulse active-low reset -> model post-reset() state.
        #1 rst_n = 1'b0;
        #1 rst_n = 1'b1;
        #1;

        fd = $fopen("vec/div_iscb.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/div_iscb.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            // vec columns: <reset> <dividend> <divisor> <q_uni> <q_bi>
            code = $fscanf(fd, "%b %b %b %b %b\n", rst_row, dd, ds, exp_uni, exp_bi);
            if (code == 5) begin
                if (rst_row) begin
                    // Mid-stream reset: re-pulse i_rst_n low to drive every
                    // register back to the model's post-reset() state from a
                    // DIRTIED state. Outputs on this row are don't-care.
                    rst_n = 1'b0;
                    #1;
                    clk = 1'b1; #1;   // posedge while held in reset
                    clk = 1'b0; #1;
                    rst_n = 1'b1;
                    #1;
                end else begin
                    // Drive this timestep's inputs and let the combinational
                    // quotient settle against the CURRENT register state.
                    dividend = dd;
                    divisor  = ds;
                    #1;
                    n = n + 1;
                    if (q_uni !== exp_uni) begin
                        $display("FAIL[uni] t=%0d dd=%b ds=%b : got %b exp %b", n - 1, dd, ds, q_uni, exp_uni);
                        fails = fails + 1;
                    end
                    if (q_bi !== exp_bi) begin
                        $display("FAIL[bi]  t=%0d dd=%b ds=%b : got %b exp %b", n - 1, dd, ds, q_bi, exp_bi);
                        fails = fails + 1;
                    end
                    // Advance state: one posedge == one forward() timestep.
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS div_iscb: %0d/%0d vectors", n, n);
        else
            $display("FAIL div_iscb: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
