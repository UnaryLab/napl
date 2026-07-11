`timescale 1ns/1ps
`default_nettype none
// GEN_DEPTH is emitted by gen/gen_square_dff.py from the op config (= the test's
// square_dff_config), so the DUT DEPTH parameter is inherited from the Python
// model. iverilog resolves this include relative to the compile cwd
// (implementation/).
`include "square_dff/vec/square_dff_params.vh"
//==============================================================================
// Self-checking testbench for square_dff.
//
// Reads golden vectors produced by gen/gen_square_dff.py (from the napl Python
// model) and asserts both polarity variants reproduce them cycle by cycle.
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line
// to decide the exit status.
//
// Vector format per line:  <rst> <i_in> <out_uni> <out_bi>
// rst=1 marks cycles where the model.reset() was replayed before driving i_in;
// the tb pulses active-low i_rst_n low across a posedge there to clear the delay
// line, proving reset equivalence from a dirtied mid-stream state.
//
// Timing model: one Python forward() timestep == one posedge i_clk. The delay
// line holds the previous inputs; the output is combinational from the current
// input and the oldest cell. So for each vector we drive i_in, let the
// combinational output settle, check it, then clock the line forward.
// i_rst_n is pulsed low first to match the Python reset() (delay line = 0).
//
// Run (from src/napl/implementation/):
//   make test OP=square_dff
//==============================================================================
module square_dff_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_in;
    wire o_out_uni, o_out_bi;

    // One module per polarity, both fed the same stimulus. DEPTH inherited from
    // the Python model via `GEN_DEPTH.
    square_dff_unipolar #(.DEPTH(`GEN_DEPTH)) dut_uni (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_in(i_in), .o_out(o_out_uni)
    );
    square_dff_bipolar #(.DEPTH(`GEN_DEPTH)) dut_bi (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_in(i_in), .o_out(o_out_bi)
    );

    integer fd, code, n, fails;
    reg rst_s, in_s, exp_uni, exp_bi;

    // Free-running clock: 10ns period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    initial begin
        i_in    = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;

        // Pulse reset low across a clock edge -> delay line = 0 (matches reset()).
        i_rst_n = 1'b0;
        @(posedge i_clk);
        #1 i_rst_n = 1'b1;

        fd = $fopen("vec/square_dff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/square_dff.vec (run `make vectors` first)");
            $finish;
        end

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_s, in_s, exp_uni, exp_bi);
            if (code == 4) begin
                // Mid-stream reset: pulse i_rst_n low across a posedge to clear
                // the delay line, mirroring the model.reset() the gen replayed.
                if (rst_s) begin
                    @(negedge i_clk);
                    i_rst_n = 1'b0;
                    @(posedge i_clk);
                    #1 i_rst_n = 1'b1;
                end
                // Drive the input in the low phase of the clock, let the
                // combinational output settle, then check before the rising
                // edge that advances the delay line.
                @(negedge i_clk);
                i_in = in_s;
                #1;
                n = n + 1;
                if (o_out_uni !== exp_uni) begin
                    $display("FAIL[uni] t=%0d i_in=%b : got %b exp %b", n, in_s, o_out_uni, exp_uni);
                    fails = fails + 1;
                end
                if (o_out_bi !== exp_bi) begin
                    $display("FAIL[bi]  t=%0d i_in=%b : got %b exp %b", n, in_s, o_out_bi, exp_bi);
                    fails = fails + 1;
                end
                // Rising edge clocks the delay line (shift in i_in).
                @(posedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS square_dff: %0d/%0d vectors", n, n);
        else
            $display("FAIL square_dff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
