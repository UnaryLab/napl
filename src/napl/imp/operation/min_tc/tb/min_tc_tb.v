// Self-checking testbench for min_tc.
// Reads vec/min_tc.vec (golden vectors from the napl Python model), applies
// each (in_0, in_1) combinationally, and compares o_output to the expected column.
// Prints "PASS" only on a full bit-exact match.
`timescale 1ns / 1ps
`default_nettype none
`include "min_tc/vec/min_tc_params.vh"


module min_tc_tb;

    reg  r_in_0;
    reg  r_in_1;
    wire w_out;

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
    localparam DATA_COLS  = 3;

    integer fd;
    integer rc;
    integer chars;
    integer n_vec;
    integer n_err;
    reg vin0, vin1, vexp;
    // The columns are scanned as text so a token outside {0, 1} always reaches
    // the guard below; each reg is wider than a legal token so an over-wide
    // token reads back over-wide instead of saturating.
    reg [8*8-1:0] tok_vin0;
    reg [8*8-1:0] tok_vin1;
    reg [8*8-1:0] tok_vexp;
    reg [8*8-1:0] extra;
    reg [8*LINE_BYTES-1:0] line;

    min_tc dut (
        .i_input_0 (r_in_0),
        .i_input_1 (r_in_1),
        .o_output  (w_out)
    );

    initial begin
        n_vec = 0;
        n_err = 0;

        fd = $fopen("vec/min_tc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/min_tc.vec");
            $finish;
        end

        // HEADER GUARD: the header is read through the same buffer as a data
        // row, so it takes the same truncation check.
        chars = $fgets(line, fd);
        if (chars == 0)
            $fatal(1, "min_tc: vec file has no header line");
        if (chars == LINE_BYTES && line[7:0] !== "\n")
            $fatal(1, "min_tc: header line fills the %0d byte line buffer", LINE_BYTES);

        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "min_tc: row %0d fills the %0d byte line buffer", n_vec + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                rc = $sscanf(line, "%s %s %s %s", tok_vin0, tok_vin1, tok_vexp, extra);
                if (rc == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "min_tc: row %0d is blank", n_vec + 1);
                end else begin
                    if (rc != DATA_COLS)
                        $fatal(1, "min_tc: row %0d scanned %0d columns, expected %0d", n_vec + 1, rc, DATA_COLS);
                    // INPUT COLUMN GUARDS: a stimulus column outside {0, 1} is a
                    // corrupt file, and is named as one rather than truncated
                    // into a bit the DUT would accept as stimulus.
                    if (tok_vin0 !== "0" && tok_vin0 !== "1")
                        $fatal(1, "min_tc: row %0d input field in_0 is \"%0s\", expected a single 0 or 1", n_vec + 1, tok_vin0);
                    vin0 = (tok_vin0 == "1");
                    if (tok_vin1 !== "0" && tok_vin1 !== "1")
                        $fatal(1, "min_tc: row %0d input field in_1 is \"%0s\", expected a single 0 or 1", n_vec + 1, tok_vin1);
                    vin1 = (tok_vin1 == "1");
                    // EXPECTED COLUMN GUARD: a golden column outside {0, 1} is a
                    // corrupt answer, and is named as one rather than truncated
                    // into a bit that can match the DUT.
                    if (tok_vexp !== "0" && tok_vexp !== "1")
                        $fatal(1, "min_tc: row %0d expected field out is \"%0s\", expected a single 0 or 1", n_vec + 1, tok_vexp);
                    vexp = (tok_vexp == "1");
                    r_in_0 = vin0;
                    r_in_1 = vin1;
                    #1; // settle combinational logic
                    if (w_out !== vexp) begin
                        n_err = n_err + 1;
                        $display("MISMATCH vec %0d: in_0=%0d in_1=%0d exp=%0d got=%0b",
                                 n_vec, vin0, vin1, vexp, w_out);
                    end
                    n_vec = n_vec + 1;
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n_vec != `GEN_VECTORS) begin
            $display("FAIL min_tc: compared %0d vectors, generator wrote %0d",
                     n_vec, `GEN_VECTORS);
            n_err = n_err + 1;
        end

        if (n_err == 0)
            $display("PASS: min_tc %0d/%0d vectors match", n_vec, n_vec);
        else
            $display("FAIL: min_tc %0d/%0d vectors mismatched", n_err, n_vec);

        $finish;
    end

endmodule

`default_nettype wire
