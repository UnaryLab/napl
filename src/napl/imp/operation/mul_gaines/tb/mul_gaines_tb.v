`timescale 1ns/1ps
`default_nettype none
`include "mul_gaines/vec/mul_gaines_params.vh"


module mul_gaines_tb;
    reg i_input_0;
    reg i_input_1;
    wire o_uni;
    wire o_bi;

    mul_gaines_unipolar dut_uni (
        .i_input_0(i_input_0),
        .i_input_1(i_input_1),
        .o_output(o_uni)
    );
    mul_gaines_bipolar dut_bi (
        .i_input_0(i_input_0),
        .i_input_1(i_input_1),
        .o_output(o_bi)
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
    localparam LINE_BYTES = 16;
    localparam DATA_COLS  = 4;

    integer fd;
    integer code;
    integer chars;
    integer count;
    integer fails;
    reg in_0_s;
    reg in_1_s;
    reg exp_uni;
    reg exp_bi;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_in_0_s;
    reg [8*8-1:0] tok_in_1_s;
    reg [8*8-1:0] tok_exp_uni;
    reg [8*8-1:0] tok_exp_bi;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        fd = $fopen("vec/mul_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_gaines.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "mul_gaines: row %0d fills the %0d byte line buffer", count + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s", tok_in_0_s, tok_in_1_s, tok_exp_uni, tok_exp_bi, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "mul_gaines: row %0d is blank", count + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "mul_gaines: row %0d scanned %0d columns, expected %0d", count + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_in_0_s !== "0" && tok_in_0_s !== "1")
                        $fatal(1, "mul_gaines: row %0d input field i_input_0 is \"%0s\", expected a single 0 or 1", count + 1, tok_in_0_s);
                    in_0_s = (tok_in_0_s == "1");
                    if (tok_in_1_s !== "0" && tok_in_1_s !== "1")
                        $fatal(1, "mul_gaines: row %0d input field i_input_1 is \"%0s\", expected a single 0 or 1", count + 1, tok_in_1_s);
                    in_1_s = (tok_in_1_s == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_uni !== "0" && tok_exp_uni !== "1")
                        $fatal(1, "mul_gaines: row %0d expected field exp_uni is \"%0s\", expected a single 0 or 1", count + 1, tok_exp_uni);
                    exp_uni = (tok_exp_uni == "1");
                    if (tok_exp_bi !== "0" && tok_exp_bi !== "1")
                        $fatal(1, "mul_gaines: row %0d expected field exp_bi is \"%0s\", expected a single 0 or 1", count + 1, tok_exp_bi);
                    exp_bi = (tok_exp_bi == "1");
                    i_input_0 = in_0_s;
                    i_input_1 = in_1_s;
                    #1;
                    count = count + 1;
                    if (o_uni !== exp_uni)
                        fails = fails + 1;
                    if (o_bi !== exp_bi)
                        fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (count != `GEN_VECTORS) begin
            $display("FAIL mul_gaines: compared %0d vectors, generator wrote %0d",
                     count, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS mul_gaines: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL mul_gaines: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
