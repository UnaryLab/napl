`timescale 1ns/1ps
`default_nettype none
`include "eq/vec/eq_params.vh"
// Golden rows: <rst_n> <in0> <in1> <out>. One eq instance takes the unipolar and
// bipolar segments in turn, the circuit being the same for both codes. Output is
// combinational (pp_delay=0), so it is checked before the posedge that advances
// the counter. rst_n=0 rows pulse reset.
// Co-sim: make test OP=eq


module eq_tb;
    reg clk, rst_n;
    reg in0, in1;
    wire out;

    eq #(.WIDTH(`GEN_WIDTH), .TOLERANCE(`GEN_TOLERANCE)) dut (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input_0(in0),
        .i_input_1(in1),
        .o_output (out)
    );

    // LINE_BYTES sizes the line buffer, which holds LINE_BYTES bytes, so a
    // newline-terminated row carries at most LINE_BYTES-1 payload bytes, and
    // DATA_COLS is the column count every data row carries. Neither constant
    // can be set wrong and still let a malformed row through, because the
    // trailing sentinel makes the column check two-sided. A DATA_COLS set too
    // large fatals on the first clean row. A LINE_BYTES set too large costs
    // guard precision rather than correctness, because the length assertion
    // can then fire only at the buffer boundary instead of near the true row
    // width, so both constants are measured from the vec file rather than
    // guessed.
    localparam LINE_BYTES = 32;
    localparam DATA_COLS  = 4;

    integer fd, code, chars, n, fails;
    reg rst_in, a, b, e;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_in;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_b;
    reg [8*8-1:0] tok_e;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        clk = 1'b0;
        in0 = 1'b0; in1 = 1'b0;

        rst_n = 1'b0;
        #1 rst_n = 1'b1;

        fd = $fopen("vec/eq.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/eq.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // One line holds one row of exactly DATA_COLS binary columns. A line
        // carrying any other column count, a line that fills the line buffer, or
        // an x or z data field ends the run with a fatal at that row, so a row
        // can never borrow a column from its neighbour. A blank line is benign
        // only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "eq: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s",
                               tok_rst_in, tok_a, tok_b, tok_e, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "eq: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "eq: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_in !== "0" && tok_rst_in !== "1")
                        $fatal(1, "eq: row %0d input field rst_in is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_in);
                    rst_in = (tok_rst_in == "1");
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "eq: row %0d input field in0 is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    if (tok_b !== "0" && tok_b !== "1")
                        $fatal(1, "eq: row %0d input field in1 is \"%0s\", expected a single 0 or 1", n + 1, tok_b);
                    b = (tok_b == "1");
                    // Expected-output column: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_e !== "0" && tok_e !== "1")
                        $fatal(1, "eq: row %0d expected field out is \"%0s\", expected a single 0 or 1", n + 1, tok_e);
                    e = (tok_e == "1");
                    if (rst_in == 1'b0) begin
                        // Mid-stream reset pulse: assert i_rst_n low across a clock
                        // edge -> counter reloads HALF. No output check.
                        rst_n = 1'b0;
                        clk = 1'b1; #1;
                        clk = 1'b0; #1;
                        rst_n = 1'b1; #1;
                    end else begin
                        in0 = a; in1 = b;
                        #1;                  // settle combinational o_output (post-update)
                        n = n + 1;
                        if (out !== e) begin
                            $display("FAIL cycle %0d: in0=%b in1=%b got %b exp %b",
                                     n - 1, a, b, out, e);
                            fails = fails + 1;
                        end
                        // advance state: one posedge consumes this cycle's inputs.
                        clk = 1'b1; #1;
                        clk = 1'b0; #1;
                    end
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL eq: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS eq: %0d/%0d vectors", n, n);
        else
            $display("FAIL eq: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
