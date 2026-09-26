`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "bi2uni/vec/bi2uni_params.vh"
// Python golden output is checked before each posedge updates acc. R requests
// an active-low reset before replay continues.
// Co-sim: make test OP=bi2uni


module bi2uni_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output;

    bi2uni #(.WIDTH(`GEN_WIDTH)) dut (
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
    // guessed.
    localparam LINE_BYTES = 32;
    localparam DATA_COLS  = 2;

    integer fd, code, chars, n, fails;
    reg [8*8-1:0] in_str, out_str, extra;
    reg in_bit, exp_out;
    reg [8*LINE_BYTES-1:0] line;

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
        i_input    = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;

        fd = $fopen("vec/bi2uni.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/bi2uni.vec (run `make vectors` first)");
            $finish;
        end

        do_reset;

        // One line holds one row of exactly DATA_COLS columns: the reset marker
        // carries R in both, a data row carries 0 or 1 in both. A line carrying
        // any other column count, a line that fills the line buffer, or a column
        // that is neither R nor 0 nor 1 ends the run with a fatal at that row, so
        // a row can never borrow a column from its neighbour. The columns are
        // scanned as characters, so that character check is what rejects an
        // unknown field here. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "bi2uni: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s", in_str, out_str, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "bi2uni: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "bi2uni: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (in_str == "R") begin
                        if (out_str !== "R")
                            $fatal(1, "bi2uni: row %0d output column is \"%0s\", expected R on a reset row", n + 1, out_str);
                        do_reset;
                    end else begin
                        if (in_str !== "0" && in_str !== "1")
                            $fatal(1, "bi2uni: row %0d field i_input is \"%0s\", expected 0 or 1 on a data row, with R in both columns on a reset row", n + 1, in_str);
                        if (out_str !== "0" && out_str !== "1")
                            $fatal(1, "bi2uni: row %0d field exp_out is \"%0s\", expected 0 or 1 on a data row, with R in both columns on a reset row", n + 1, out_str);
                        in_bit  = (in_str == "1");
                        exp_out = (out_str == "1");
                        i_input = in_bit;
                        #1;                 // let the combinational output settle
                        n = n + 1;
                        if (o_output !== exp_out) begin
                            $display("FAIL cyc=%0d i_input=%b : got %b exp %b", n, in_bit, o_output, exp_out);
                            fails = fails + 1;
                        end
                        #1 i_clk = 1'b1; #1 i_clk = 1'b0; #1;
                    end
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL bi2uni: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS bi2uni: %0d/%0d vectors", n, n);
        else
            $display("FAIL bi2uni: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
