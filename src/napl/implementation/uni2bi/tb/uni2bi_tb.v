`timescale 1ns/1ps
`default_nettype none
// GEN_WIDTH is emitted by gen/gen_uni2bi.py from the op config (= the test's
// uni2bi_config), so the DUT parameter is inherited from the Python model.
// iverilog resolves this include relative to the compile cwd (implementation/).
`include "uni2bi/vec/uni2bi_params.vh"
//==============================================================================
// Self-checking testbench for uni2bi.
//
// Reads golden vectors produced by gen/gen_uni2bi.py (from the napl Python
// model) and replays them cycle by cycle. uni2bi is a Mealy machine: o_out is
// combinational in i_in and the accumulator, which advances on each posedge
// i_clk. So per cycle we drive i_in, let o_out settle, check it, then clock
// once to advance the accumulator. i_rst_n is pulsed low first to start from
// the model's post-reset() state (acc = 0).
//
// A bare "R" line in the vector file marks a MID-STREAM reset: the generator
// called model.reset() there, so the tb pulses i_rst_n low to reproduce it from
// a dirtied accumulator, proving reset equivalence beyond t=0.
//
// Prints "PASS ..." iff every vector matches; the Makefile greps for that line
// to decide the exit status.
//
// Run (from src/napl/implementation/):
//   make test OP=uni2bi
//==============================================================================
module uni2bi_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_in;
    wire o_out;

    uni2bi #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_in    (i_in),
        .o_out   (o_out)
    );

    integer fd, code, n, fails;
    reg [127:0] tok;
    reg in_bit, exp_out;

    // Pulse i_rst_n low across a clock edge to clear acc (Python reset()).
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

        fd = $fopen("vec/uni2bi.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/uni2bi.vec (run `make vectors` first)");
            $finish;
        end

        // Start from the model's post-reset() state (acc = 0).
        do_reset;

        while (!$feof(fd)) begin
            // Each line is either "R" (mid-stream reset) or "<in> <out>".
            code = $fscanf(fd, "%s", tok);
            if (code == 1) begin
                if (tok == "R") begin
                    do_reset;
                end else begin
                    // tok holds the input bit; read the expected output next.
                    in_bit  = (tok == "1");
                    code    = $fscanf(fd, "%b", exp_out);
                    i_in    = in_bit;
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
            $display("PASS uni2bi: %0d/%0d vectors", n, n);
        else
            $display("FAIL uni2bi: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
