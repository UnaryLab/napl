`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH/WIDTH mirror the Python model configuration.
`include "div_cordiv/vec/div_cordiv_params.vh"
// Python golden output is checked before each posedge advances buffer and index.
// R requests an active-low reset before replay continues.
// Co-sim: make test OP=div_cordiv


module div_cordiv_tb;
    reg  clk, rst_n;
    reg  dividend, divisor;
    wire quotient;

    div_cordiv #(
        .DEPTH(`GEN_DEPTH),
        .WIDTH(`GEN_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_dividend(dividend),
        .i_divisor(divisor),
        .o_output(quotient)
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
    reg [8*8-1:0] tag, extra;
    reg [8*8-1:0] tok_ds;
    reg [8*8-1:0] tok_exp_q;
    reg dv, ds, exp_q;
    reg [8*LINE_BYTES-1:0] line;


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
        if (`GEN_PP_DELAY != 0) begin
            $display(
                "FAIL div_cordiv: observed latency 0, expected pp_delay %0d",
                `GEN_PP_DELAY
            );
            $finish;
        end

        do_reset;

        fd = $fopen("vec/div_cordiv.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/div_cordiv.vec (run `make vectors` first)");
            $finish;
        end

        // One line holds one row: either the reset sentinel "R" alone, or a tag
        // column of 0 or 1 followed by exactly DATA_COLS-1 binary data columns. A
        // line carrying any other column count, a line that fills the line buffer,
        // a data row whose tag is not 0 or 1, or an x or z data field ends the run
        // with a fatal at that row, so a row can never borrow a column from its
        // neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            // Rows are "<dividend> <divisor> <quotient>" or the reset sentinel "R".
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "div_cordiv: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tag, tok_ds, tok_exp_q, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "div_cordiv: row %0d is blank", n + 1);
                end else if (code == 1 && tag == "R") begin
                    do_reset;
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "div_cordiv: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (tag !== "0" && tag !== "1")
                        $fatal(1, "div_cordiv: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with R alone on its own line", n + 1, tag);
                    dv = (tag == "1");
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_ds !== "0" && tok_ds !== "1")
                        $fatal(1, "div_cordiv: row %0d input field divisor is \"%0s\", expected a single 0 or 1", n + 1, tok_ds);
                    ds = (tok_ds == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_q !== "0" && tok_exp_q !== "1")
                        $fatal(1, "div_cordiv: row %0d expected field exp_q is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_q);
                    exp_q = (tok_exp_q == "1");
                    dividend = dv;
                    divisor  = ds;
                    #1;                  // let the combinational quotient settle
                    n = n + 1;
                    if (quotient !== exp_q) begin
                        $display("FAIL cyc=%0d dividend=%b divisor=%b : got %b exp %b",
                                 n, dv, ds, quotient, exp_q);
                        fails = fails + 1;
                    end
                    clk = 1'b1; #1;
                    clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL div_cordiv: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS div_cordiv: %0d/%0d vectors", n, n);
        else
            $display("FAIL div_cordiv: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
