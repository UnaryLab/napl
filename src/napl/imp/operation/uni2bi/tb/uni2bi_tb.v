`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "uni2bi/vec/uni2bi_params.vh"
// Python golden output is checked before each posedge updates acc. R requests
// active-low reset before replay continues.
// Co-sim: make test OP=uni2bi


module uni2bi_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output;

    uni2bi #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input    (i_input),
        .o_output   (o_output)
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
    // guessed. Reset rows are the rows that do not carry DATA_COLS columns:
    // each holds its tag alone and is accepted ahead of the column check.
    localparam LINE_BYTES = 32;
    localparam DATA_COLS  = 2;

    integer fd, code, chars, n, fails;
    reg [127:0] tok;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_exp_out;
    reg [8*LINE_BYTES-1:0] line;
    reg in_bit, exp_out;

    // Reset clears acc across a clock edge.

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
        i_input    = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;

        fd = $fopen("vec/uni2bi.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/uni2bi.vec (run `make vectors` first)");
            $finish;
        end

        do_reset;

        // A line carrying any other column count, a line that fills the line
        // buffer, a data row whose tag is not 0 or 1, or an x or z data field
        // ends the run with a fatal at that row, so a row can never borrow a
        // column from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "uni2bi: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s", tok, tok_exp_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "uni2bi: row %0d is blank", n + 1);
                end else if (code == 1 && tok == "R") begin
                    do_reset;
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "uni2bi: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (tok !== "0" && tok !== "1")
                        $fatal(1, "uni2bi: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with R alone on its own line", n + 1, tok);
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "uni2bi: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    in_bit  = (tok == "1");
                    i_input    = in_bit;
                    #1;
                    n = n + 1;
                    if (o_output !== exp_out) begin
                        $display("FAIL cyc=%0d i_input=%b : got %b exp %b", n, in_bit, o_output, exp_out);
                        fails = fails + 1;
                    end
                    #1 i_clk = 1'b1; #1 i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL uni2bi: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS uni2bi: %0d/%0d vectors", n, n);
        else
            $display("FAIL uni2bi: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
