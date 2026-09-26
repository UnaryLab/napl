`timescale 1ns/1ps
`default_nettype none
`include "relu_tc/vec/relu_tc_params.vh"

// Self-checking co-simulation testbench for relu_tc. The output is combinational
// in the arrival cycle, so it is checked before the clock edge advances the
// internal reference counter.

module relu_tc_tb;
    reg i_clk, i_rst_n, i_input;
    wire o_output;

    relu_tc #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output)
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
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg rst, expected;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst;
    reg [8*8-1:0] tok_i_input;
    reg [8*8-1:0] tok_expected;
    reg [8*LINE_BYTES-1:0] line;

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
        i_rst_n = 1'b1;
        i_input = 1'b0;
        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL relu_tc: expected pp_delay 0");
            $finish;
        end
        reset_dut;

        fd = $fopen("vec/relu_tc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/relu_tc.vec");
            $finish;
        end
        // A line carrying any other column count, a line that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "relu_tc: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok_rst, tok_i_input, tok_expected, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "relu_tc: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "relu_tc: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "relu_tc: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    if (tok_i_input !== "0" && tok_i_input !== "1")
                        $fatal(1, "relu_tc: row %0d input field i_input is \"%0s\", expected a single 0 or 1", n + 1, tok_i_input);
                    i_input = (tok_i_input == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_expected !== "0" && tok_expected !== "1")
                        $fatal(1, "relu_tc: row %0d expected field expected is \"%0s\", expected a single 0 or 1", n + 1, tok_expected);
                    expected = (tok_expected == "1");
                    if (rst)
                        reset_dut;
                    #1;
                    n = n + 1;
                    if (o_output !== expected) begin
                        $display("FAIL cycle=%0d in=%b got=%b exp=%b", n, i_input, o_output, expected);
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
        if (n != `GEN_VECTORS) begin
            $display("FAIL relu_tc: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS relu_tc: %0d/%0d vectors", n, n);
        else
            $display("FAIL relu_tc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
