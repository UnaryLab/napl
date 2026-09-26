`timescale 1ns/1ps
`default_nettype none
`include "sqrt_tracejkff/vec/sqrt_tracejkff_params.vh"
// Python golden outputs must be checked before the committing posedge; checking
// later advances trace one cycle early after reset. R clears trace and acc.
// Co-sim: make test OP=sqrt_tracejkff


module sqrt_tracejkff_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bip;

    sqrt_tracejkff_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_output   (out_uni)
    );

    sqrt_tracejkff_bipolar dut_bip (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_output   (out_bip)
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
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg a, exp_u, exp_b;
    reg [8*8-1:0] tok, extra;
    reg [8*8-1:0] tok_exp_u;
    reg [8*8-1:0] tok_exp_b;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Reset loads trace=0 and acc=0.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_tracejkff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_tracejkff.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // A line carrying any other column count, a line that fills the line
        // buffer, a data row whose tag is not 0 or 1, or an x or z data field
        // ends the run with a fatal at that row, so a row can never borrow a
        // column from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "sqrt_tracejkff: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok, tok_exp_u, tok_exp_b, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "sqrt_tracejkff: row %0d is blank", n + 1);
                end else if (code == 1 && tok == "R") begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "sqrt_tracejkff: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (tok !== "0" && tok !== "1")
                        $fatal(1, "sqrt_tracejkff: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with R alone on its own line", n + 1, tok);
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_u !== "0" && tok_exp_u !== "1")
                        $fatal(1, "sqrt_tracejkff: row %0d expected field exp_u is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_u);
                    exp_u = (tok_exp_u == "1");
                    if (tok_exp_b !== "0" && tok_exp_b !== "1")
                        $fatal(1, "sqrt_tracejkff: row %0d expected field exp_b is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_b);
                    exp_b = (tok_exp_b == "1");
                    a = (tok == "1");
                    n = n + 1;
                    in_bit = a;
                    #1;
                    if (out_uni !== exp_u) begin
                        $display("FAIL uni cyc=%0d in=%b : got %b exp %b", n, a, out_uni, exp_u);
                        fails = fails + 1;
                    end
                    if (out_bip !== exp_b) begin
                        $display("FAIL bip cyc=%0d in=%b : got %b exp %b", n, a, out_bip, exp_b);
                        fails = fails + 1;
                    end
                    @(posedge clk);
                    @(negedge clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL sqrt_tracejkff: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS sqrt_tracejkff: %0d/%0d vectors", n, n);
        else
            $display("FAIL sqrt_tracejkff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
