`timescale 1ns/1ps
`default_nettype none
`include "tanh_pn/vec/tanh_pn_params.vh"


module tanh_pn_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_input;
    wire o_output;

    tanh_pn #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input),
        .o_output(o_output)
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

    integer fd;
    integer code;
    integer chars;
    integer count;
    integer fails;
    reg [8*8-1:0] token, extra;
    reg [8*8-1:0] tok_expected;
    reg [8*LINE_BYTES-1:0] line;
    reg expected;

    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;


    task reset_dut;
        begin
            i_rst_n = 1'b0;
            @(posedge i_clk);
            @(negedge i_clk);
            i_rst_n = 1'b1;
        end
    endtask


    initial begin
        i_input = 1'b0;
        i_rst_n = 1'b1;
        reset_dut;

        if (`GEN_PP_DELAY != 1) begin
            $display("ERROR: tanh_pn pp_delay must be 1");
            $finish;
        end

        fd = $fopen("vec/tanh_pn.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/tanh_pn.vec");
            $finish;
        end

        count = 0;
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
                    $fatal(1, "tanh_pn: row %0d fills the %0d byte line buffer", count + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s", token, tok_expected, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "tanh_pn: row %0d is blank", count + 1);
                end else if (code == 1 && token == "R") begin
                    reset_dut;
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "tanh_pn: row %0d scanned %0d columns, expected %0d", count + 1, code, DATA_COLS);
                    if (token !== "0" && token !== "1")
                        $fatal(1, "tanh_pn: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with R alone on its own line", count + 1, token);
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_expected !== "0" && tok_expected !== "1")
                        $fatal(1, "tanh_pn: row %0d expected field expected is \"%0s\", expected a single 0 or 1", count + 1, tok_expected);
                    expected = (tok_expected == "1");
                    i_input = (token[7:0] == "1");
                    #1;
                    count = count + 1;
                    if (o_output !== expected) begin
                        $display(
                            "FAIL tanh_pn cycle %0d: in=%b got=%b expected=%b",
                            count, i_input, o_output, expected
                        );
                        fails = fails + 1;
                    end
                    @(posedge i_clk);
                    @(negedge i_clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (count != `GEN_VECTORS) begin
            $display("FAIL tanh_pn: compared %0d vectors, generator wrote %0d",
                     count, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS tanh_pn: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL tanh_pn: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
