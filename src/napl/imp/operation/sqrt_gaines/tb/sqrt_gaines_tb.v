`timescale 1ns/1ps
`default_nettype none
`include "sqrt_gaines/vec/sqrt_gaines_params.vh"


module sqrt_gaines_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_input_uni;
    reg i_input_bi;
    wire o_output_uni;
    wire o_output_bi;

    sqrt_gaines_unipolar #(.WIDTH(`GEN_WIDTH)) dut_uni (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input_uni),
        .o_output(o_output_uni)
    );
    sqrt_gaines_bipolar #(.WIDTH(`GEN_WIDTH)) dut_bi (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input_bi),
        .o_output(o_output_bi)
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
    localparam DATA_COLS  = 5;

    integer fd;
    integer code;
    integer chars;
    integer count;
    integer fails;
    reg reset_flag;
    reg expected_uni;
    reg expected_bi;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_reset_flag;
    reg [8*8-1:0] tok_i_input_uni;
    reg [8*8-1:0] tok_expected_uni;
    reg [8*8-1:0] tok_i_input_bi;
    reg [8*8-1:0] tok_expected_bi;
    reg [8*LINE_BYTES-1:0] line;

    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;


    task reset_duts;
        begin
            i_rst_n = 1'b0;
            @(posedge i_clk);
            @(negedge i_clk);
            i_rst_n = 1'b1;
        end
    endtask


    initial begin
        i_rst_n = 1'b1;
        i_input_uni = 1'b0;
        i_input_bi = 1'b0;

        if (`GEN_PP_DELAY != 1) begin
            $display("ERROR: sqrt_gaines pp_delay must be 1");
            $finish;
        end

        fd = $fopen("vec/sqrt_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_gaines.vec");
            $finish;
        end

        count = 0;
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
                    $fatal(1, "sqrt_gaines: row %0d fills the %0d byte line buffer", count + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s",
                               tok_reset_flag, tok_i_input_uni, tok_expected_uni,
                               tok_i_input_bi, tok_expected_bi, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "sqrt_gaines: row %0d is blank", count + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "sqrt_gaines: row %0d scanned %0d columns, expected %0d", count + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_reset_flag !== "0" && tok_reset_flag !== "1")
                        $fatal(1, "sqrt_gaines: row %0d input field reset_flag is \"%0s\", expected a single 0 or 1", count + 1, tok_reset_flag);
                    reset_flag = (tok_reset_flag == "1");
                    if (tok_i_input_uni !== "0" && tok_i_input_uni !== "1")
                        $fatal(1, "sqrt_gaines: row %0d input field i_input_uni is \"%0s\", expected a single 0 or 1", count + 1, tok_i_input_uni);
                    i_input_uni = (tok_i_input_uni == "1");
                    if (tok_i_input_bi !== "0" && tok_i_input_bi !== "1")
                        $fatal(1, "sqrt_gaines: row %0d input field i_input_bi is \"%0s\", expected a single 0 or 1", count + 1, tok_i_input_bi);
                    i_input_bi = (tok_i_input_bi == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_expected_uni !== "0" && tok_expected_uni !== "1")
                        $fatal(1, "sqrt_gaines: row %0d expected field expected_uni is \"%0s\", expected a single 0 or 1", count + 1, tok_expected_uni);
                    expected_uni = (tok_expected_uni == "1");
                    if (tok_expected_bi !== "0" && tok_expected_bi !== "1")
                        $fatal(1, "sqrt_gaines: row %0d expected field expected_bi is \"%0s\", expected a single 0 or 1", count + 1, tok_expected_bi);
                    expected_bi = (tok_expected_bi == "1");
                    if (reset_flag)
                        reset_duts;
                    #1;
                    count = count + 1;
                    if (o_output_uni !== expected_uni) begin
                        $display(
                            "FAIL sqrt_gaines unipolar cycle %0d: got=%b expected=%b",
                            count, o_output_uni, expected_uni
                        );
                        fails = fails + 1;
                    end
                    if (o_output_bi !== expected_bi) begin
                        $display(
                            "FAIL sqrt_gaines bipolar cycle %0d: got=%b expected=%b",
                            count, o_output_bi, expected_bi
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
            $display("FAIL sqrt_gaines: compared %0d vectors, generator wrote %0d",
                     count, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS sqrt_gaines: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL sqrt_gaines: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
