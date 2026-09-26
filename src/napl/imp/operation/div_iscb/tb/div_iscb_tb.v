`timescale 1ns/1ps
`default_nettype none
`include "div_iscb/vec/div_iscb_params.vh"
// Python golden outputs for both polarities are checked before each posedge
// advances state. A reset row pulses active-low reset and has don't-care outputs.
// Co-sim: make test OP=div_iscb


module div_iscb_tb;
    reg  clk;
    reg  rst_n;
    reg  dividend, divisor;
    reg  b2u_in;

    wire q_uni, q_bi, b2u_out;

    div_iscb_unipolar dut_uni (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_dividend(dividend), .i_divisor(divisor),
        .o_output(q_uni)
    );
    div_iscb_bipolar dut_bi (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_dividend(dividend), .i_divisor(divisor),
        .o_output(q_bi)
    );

    // The bi2uni helper's low clamp is unreachable through div_iscb_bipolar, so
    // a second copy is driven on its own stimulus column at the same width the
    // bipolar variant elaborates.
    bi2uni #(.WIDTH(3)) dut_b2u (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_input(b2u_in), .o_output(b2u_out)
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
    localparam DATA_COLS  = 7;

    integer fd, code, chars, n, fails;
    reg rst_row;
    reg dd, ds, exp_uni, exp_bi, bx, exp_b2u;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_row;
    reg [8*8-1:0] tok_dd, tok_ds, tok_exp_uni, tok_exp_bi, tok_bx, tok_exp_b2u;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        clk = 1'b0;
        rst_n = 1'b1;
        dividend = 1'b0;
        divisor = 1'b0;
        b2u_in = 1'b0;

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
        // One line holds one row of exactly DATA_COLS binary columns. A line
        // carrying any other column count, a line that fills the line buffer, or
        // an x or z data field on a data row ends the run with a fatal at that
        // row, so a row can never borrow a column from its neighbour. The reset
        // marker row carries x in its three output columns, and its three stimulus
        // columns are never driven into the DUTs, so the run compares nothing on
        // that row and checks only its reset column. A blank line is benign only
        // at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "div_iscb: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // vec columns: <reset> <dividend> <divisor> <q_uni> <q_bi> <b2u_in> <b2u_out>
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s %s %s",
                               tok_rst_row, tok_dd, tok_ds, tok_exp_uni, tok_exp_bi,
                               tok_bx, tok_exp_b2u, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "div_iscb: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "div_iscb: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_row !== "0" && tok_rst_row !== "1")
                        $fatal(1, "div_iscb: row %0d input field rst_row is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_row);
                    rst_row = (tok_rst_row == "1");
                    if (rst_row === 1'b0) begin
                        if (tok_dd !== "0" && tok_dd !== "1")
                            $fatal(1, "div_iscb: row %0d input field dividend is \"%0s\", expected a single 0 or 1", n + 1, tok_dd);
                        dd = (tok_dd == "1");
                        if (tok_ds !== "0" && tok_ds !== "1")
                            $fatal(1, "div_iscb: row %0d input field divisor is \"%0s\", expected a single 0 or 1", n + 1, tok_ds);
                        ds = (tok_ds == "1");
                        if (tok_bx !== "0" && tok_bx !== "1")
                            $fatal(1, "div_iscb: row %0d input field b2u_in is \"%0s\", expected a single 0 or 1", n + 1, tok_bx);
                        bx = (tok_bx == "1");
                        // Expected-output columns: a token wider than one bit would be
                        // truncated into a bit that can match the DUT, so a wrong answer
                        // would be accepted. Compare the column as characters instead.
                        if (tok_exp_uni !== "0" && tok_exp_uni !== "1")
                            $fatal(1, "div_iscb: row %0d expected field exp_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni);
                        exp_uni = (tok_exp_uni == "1");
                        if (tok_exp_bi !== "0" && tok_exp_bi !== "1")
                            $fatal(1, "div_iscb: row %0d expected field exp_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi);
                        exp_bi = (tok_exp_bi == "1");
                        if (tok_exp_b2u !== "0" && tok_exp_b2u !== "1")
                            $fatal(1, "div_iscb: row %0d expected field exp_b2u is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_b2u);
                        exp_b2u = (tok_exp_b2u == "1");
                    end
                    if (rst_row) begin
                        rst_n = 1'b0;
                        #1;
                        clk = 1'b1; #1;   // posedge while held in reset
                        clk = 1'b0; #1;
                        rst_n = 1'b1;
                        #1;
                    end else begin
                        dividend = dd;
                        divisor  = ds;
                        b2u_in   = bx;
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
                        if (b2u_out !== exp_b2u) begin
                            $display("FAIL[b2u] t=%0d in=%b : got %b exp %b", n - 1, bx, b2u_out, exp_b2u);
                            fails = fails + 1;
                        end
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
            $display("FAIL div_iscb: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS div_iscb: %0d/%0d vectors", n, n);
        else
            $display("FAIL div_iscb: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
