`timescale 1ns/1ps
`default_nettype none
`include "sqrt_traceiscb/vec/sqrt_traceiscb_params.vh"
// Python golden outputs for both polarities are checked before each posedge
// advances state. The reset column requests active-low reset before its row.
// Co-sim: make test OP=sqrt_traceiscb


module sqrt_traceiscb_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bi;

    sqrt_traceiscb_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_output   (out_uni)
    );

    sqrt_traceiscb_bipolar dut_bi (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_output   (out_bi)
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
    reg a, exp_uni, exp_bi, rst_mark;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_mark;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_exp_uni;
    reg [8*8-1:0] tok_exp_bi;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Release reset on the negedge immediately before the first checked row.
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/sqrt_traceiscb.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_traceiscb.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // A line carrying any other column count, a line that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "sqrt_traceiscb: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s", tok_rst_mark, tok_a, tok_exp_uni, tok_exp_bi, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "sqrt_traceiscb: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "sqrt_traceiscb: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_mark !== "0" && tok_rst_mark !== "1")
                        $fatal(1, "sqrt_traceiscb: row %0d input field rst_mark is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_mark);
                    rst_mark = (tok_rst_mark == "1");
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "sqrt_traceiscb: row %0d input field a is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_uni !== "0" && tok_exp_uni !== "1")
                        $fatal(1, "sqrt_traceiscb: row %0d expected field exp_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni);
                    exp_uni = (tok_exp_uni == "1");
                    if (tok_exp_bi !== "0" && tok_exp_bi !== "1")
                        $fatal(1, "sqrt_traceiscb: row %0d expected field exp_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi);
                    exp_bi = (tok_exp_bi == "1");
                    if (rst_mark) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    n = n + 1;
                    in_bit = a;
                    #1;
                    if (out_uni !== exp_uni) begin
                        $display("FAIL uni cyc=%0d in=%b : got %b exp %b", n, a, out_uni, exp_uni);
                        fails = fails + 1;
                    end
                    if (out_bi !== exp_bi) begin
                        $display("FAIL bi  cyc=%0d in=%b : got %b exp %b", n, a, out_bi, exp_bi);
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
            $display("FAIL sqrt_traceiscb: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS sqrt_traceiscb: %0d/%0d vectors (uni+bi)", n, n);
        else
            $display("FAIL sqrt_traceiscb: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
