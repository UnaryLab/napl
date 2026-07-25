`timescale 1ns/1ps
`default_nettype none
// GEN_WIDTH is emitted by gen/gen_bi2uni.py from the op config (= the test's
// bi2uni_config), so the DUT parameter is inherited from the Python model.
// iverilog resolves this include relative to the compile cwd (imp/).
`include "bi2uni/vec/bi2uni_params.vh"
//==============================================================================
// Self-checking testbench for bi2uni.
//
// Reads golden vectors produced by gen/gen_bi2uni.py (from the napl Python
// model) and replays them cycle by cycle. bi2uni is a Mealy machine: o_out is
// combinational in i_in and the accumulator, which advances on each posedge
// i_clk. So per cycle we drive i_in, let o_out settle, check it, then clock
// once to advance the accumulator. i_rst_n is pulsed low first to start from
// the model's post-reset() state (acc = 0).
//
// A vector line whose fields are the marker "R" is a MID-STREAM reset request:
// the tb pulses i_rst_n low (reproducing the model's reset() from a dirtied
// accumulator) before resuming, proving reset equivalence away from t=0.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line.
//
// Run (from src/napl/imp/):
//   make test OP=bi2uni
//==============================================================================
module bi2uni_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_in;
    wire o_out;

    bi2uni #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_in    (i_in),
        .o_out   (o_out)
    );

    integer fd, code, n, fails;
    reg [8*8-1:0] in_str, out_str;
    reg in_bit, exp_out;

    // pulse active-low reset to load acc = 0 (post-reset() state).
    task do_reset;
        begin
            i_rst_n = 1'b0;
            #1 i_clk = 1'b1; #1 i_clk = 1'b0;
            i_rst_n = 1'b1;
            #1;
        end
    endtask

    initial begin
        i_clk   = 1'b0;
        i_in    = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;

        fd = $fopen("vec/bi2uni.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/bi2uni.vec (run `make vectors` first)");
            $finish;
        end

        do_reset;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s %s\n", in_str, out_str);
            if (code == 2) begin
                if (in_str == "R") begin
                    // mid-stream reset request: restart from the model's
                    // post-reset() state with a dirtied accumulator behind us.
                    do_reset;
                end else begin
                    in_bit  = (in_str == "1");
                    exp_out = (out_str == "1");
                    i_in = in_bit;
                    #1;                 // let the combinational output settle
                    n = n + 1;
                    if (o_out !== exp_out) begin
                        $display("FAIL cyc=%0d i_in=%b : got %b exp %b", n, in_bit, o_out, exp_out);
                        fails = fails + 1;
                    end
                    // advance the accumulator one step.
                    #1 i_clk = 1'b1; #1 i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS bi2uni: %0d/%0d vectors", n, n);
        else
            $display("FAIL bi2uni: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
