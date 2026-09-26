`timescale 1ns/1ps
`default_nettype none
`include "pow_delay/vec/pow_delay_params.vh"
// Co-sim: make test OP=pow_delay


module pow_delay_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output_uni, o_output_bi;
    wire o_output_deep_uni, o_output_deep_bi;

    pow_delay_unipolar #(.N(`GEN_N), .DEPTH(`GEN_DEPTH)) dut_uni (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_input), .o_output(o_output_uni)
    );
    pow_delay_bipolar #(.N(`GEN_N), .DEPTH(`GEN_DEPTH)) dut_bi (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_input), .o_output(o_output_bi)
    );
    pow_delay_unipolar #(.N(`GEN_N), .DEPTH(`GEN_DEEP_DEPTH)) dut_deep_uni (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_input), .o_output(o_output_deep_uni)
    );
    pow_delay_bipolar #(.N(`GEN_N), .DEPTH(`GEN_DEEP_DEPTH)) dut_deep_bi (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_input), .o_output(o_output_deep_bi)
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
    localparam LINE_BYTES = 128;
    localparam DATA_COLS  = 6;

    integer fd, code, chars, vectors, fails;
    reg rst_s, input_s, expected_uni, expected_bi;
    reg expected_deep_uni, expected_deep_bi;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_s;
    reg [8*8-1:0] tok_input_s;
    reg [8*8-1:0] tok_expected_uni;
    reg [8*8-1:0] tok_expected_bi;
    reg [8*8-1:0] tok_expected_deep_uni;
    reg [8*8-1:0] tok_expected_deep_bi;
    reg [8*LINE_BYTES-1:0] line;

    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    initial begin
        i_input = 1'b0;
        i_rst_n = 1'b1;
        vectors = 0;
        fails = 0;

        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL pow_delay: expected pp_delay 0, got %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        i_rst_n = 1'b0;
        @(posedge i_clk);
        #1 i_rst_n = 1'b1;

        fd = $fopen("vec/pow_delay.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/pow_delay.vec (run `make vectors` first)");
            $finish;
        end

        // One line holds one row of exactly 6 binary columns. A line carrying any
        // other column count, a line that fills the line buffer, or an x or z data
        // field ends the run with a fatal at that row, so a row can never borrow a
        // column from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "pow_delay: row %0d fills the %0d byte line buffer", vectors + 1, LINE_BYTES);
                code = $sscanf(line, "%s %s %s %s %s %s %s", tok_rst_s, tok_input_s,
                               tok_expected_uni, tok_expected_bi,
                               tok_expected_deep_uni, tok_expected_deep_bi, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "pow_delay: row %0d is blank", vectors + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "pow_delay: row %0d scanned %0d columns, expected %0d", vectors + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_s !== "0" && tok_rst_s !== "1")
                        $fatal(1, "pow_delay: row %0d input field rst_s is \"%0s\", expected a single 0 or 1", vectors + 1, tok_rst_s);
                    rst_s = (tok_rst_s == "1");
                    if (tok_input_s !== "0" && tok_input_s !== "1")
                        $fatal(1, "pow_delay: row %0d input field input_s is \"%0s\", expected a single 0 or 1", vectors + 1, tok_input_s);
                    input_s = (tok_input_s == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_expected_uni !== "0" && tok_expected_uni !== "1")
                        $fatal(1, "pow_delay: row %0d expected field expected_uni is \"%0s\", expected a single 0 or 1", vectors + 1, tok_expected_uni);
                    expected_uni = (tok_expected_uni == "1");
                    if (tok_expected_bi !== "0" && tok_expected_bi !== "1")
                        $fatal(1, "pow_delay: row %0d expected field expected_bi is \"%0s\", expected a single 0 or 1", vectors + 1, tok_expected_bi);
                    expected_bi = (tok_expected_bi == "1");
                    if (tok_expected_deep_uni !== "0" && tok_expected_deep_uni !== "1")
                        $fatal(1, "pow_delay: row %0d expected field expected_deep_uni is \"%0s\", expected a single 0 or 1", vectors + 1, tok_expected_deep_uni);
                    expected_deep_uni = (tok_expected_deep_uni == "1");
                    if (tok_expected_deep_bi !== "0" && tok_expected_deep_bi !== "1")
                        $fatal(1, "pow_delay: row %0d expected field expected_deep_bi is \"%0s\", expected a single 0 or 1", vectors + 1, tok_expected_deep_bi);
                    expected_deep_bi = (tok_expected_deep_bi == "1");
                    if (rst_s) begin
                        @(negedge i_clk);
                        i_rst_n = 1'b0;
                        @(posedge i_clk);
                        #1 i_rst_n = 1'b1;
                    end
                    @(negedge i_clk);
                    i_input = input_s;
                    #1;
                    vectors = vectors + 1;
                    if (o_output_uni !== expected_uni) begin
                        $display("FAIL[uni] t=%0d input=%b got=%b expected=%b",
                                 vectors, input_s, o_output_uni, expected_uni);
                        fails = fails + 1;
                    end
                    if (o_output_bi !== expected_bi) begin
                        $display("FAIL[bi] t=%0d input=%b got=%b expected=%b",
                                 vectors, input_s, o_output_bi, expected_bi);
                        fails = fails + 1;
                    end
                    if (o_output_deep_uni !== expected_deep_uni) begin
                        $display("FAIL[deep uni] t=%0d input=%b got=%b expected=%b",
                                 vectors, input_s, o_output_deep_uni, expected_deep_uni);
                        fails = fails + 1;
                    end
                    if (o_output_deep_bi !== expected_deep_bi) begin
                        $display("FAIL[deep bi] t=%0d input=%b got=%b expected=%b",
                                 vectors, input_s, o_output_deep_bi, expected_deep_bi);
                        fails = fails + 1;
                    end
                    @(posedge i_clk);
                end
            end
        end
        $fclose(fd);

        if (vectors != `GEN_VECTORS) begin
            $display("FAIL pow_delay: consumed %0d rows, generator emitted %0d",
                     vectors, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS pow_delay: %0d/%0d vectors", vectors, vectors);
        else
            $display("FAIL pow_delay: %0d mismatch(es) over %0d vectors",
                     fails, vectors);
        $finish;
    end
endmodule
`default_nettype wire
