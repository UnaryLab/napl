`timescale 1ns/1ps
`default_nettype none
`include "add_gaines/vec/add_gaines_params.vh"


module add_gaines_tb;
    reg i_clk;
    reg i_rst_n;
    reg [`GEN_SCALED_ENTRY-1:0] i_scaled_uni;
    reg [`GEN_SCALED_ENTRY-1:0] i_scaled_bi;
    reg [`GEN_UNSCALED_ENTRY-1:0] i_unscaled;
    wire o_scaled_uni;
    wire o_scaled_bi;
    wire o_unscaled;

    add_gaines #(
        .SCALED(`GEN_SCALED),
        .ENTRY(`GEN_SCALED_ENTRY),
        .SELECT_WIDTH(`GEN_SCALED_SELECT_WIDTH)
    ) dut_scaled_uni (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_scaled_uni),
        .o_output(o_scaled_uni)
    );
    add_gaines #(
        .SCALED(`GEN_SCALED),
        .ENTRY(`GEN_SCALED_ENTRY),
        .SELECT_WIDTH(`GEN_SCALED_SELECT_WIDTH)
    ) dut_scaled_bi (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_scaled_bi),
        .o_output(o_scaled_bi)
    );
    add_gaines #(
        .SCALED(`GEN_UNSCALED),
        .ENTRY(`GEN_UNSCALED_ENTRY),
        .SELECT_WIDTH(`GEN_UNSCALED_SELECT_WIDTH)
    ) dut_unscaled (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_unscaled),
        .o_output(o_unscaled)
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
    localparam LINE_BYTES = 64;
    localparam DATA_COLS  = 7;

    localparam integer MAX_CHARS = (`GEN_SCALED_ENTRY > `GEN_UNSCALED_ENTRY ? `GEN_SCALED_ENTRY : `GEN_UNSCALED_ENTRY) + 1;

    integer fd;
    integer code;
    integer chars;
    integer count;
    integer fails;
    reg reset_flag;
    reg expected_scaled_uni;
    reg expected_scaled_bi;
    reg expected_unscaled;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_reset_flag;
    reg [8*8-1:0] tok_expected_scaled_uni;
    reg [8*8-1:0] tok_expected_scaled_bi;
    reg [8*8-1:0] tok_expected_unscaled;
    reg [MAX_CHARS*8-1:0] tok_i_scaled_uni;
    reg [MAX_CHARS*8-1:0] tok_i_scaled_bi;
    reg [MAX_CHARS*8-1:0] tok_i_unscaled;
    reg [8*LINE_BYTES-1:0] line;

    // Characters $sscanf("%s") stored: the bit width the golden row carries.
    function integer token_len;
        input [MAX_CHARS*8-1:0] token;
        integer index;
        begin
            token_len = 0;
            for (index = 0; index < MAX_CHARS; index = index + 1)
                if (token[index*8 +: 8] != 8'h00)
                    token_len = token_len + 1;
        end
    endfunction


    // 1 when the column is not exactly width characters, each of them 0 or 1.
    function token_bad;
        input [MAX_CHARS*8-1:0] token;
        input integer width;
        integer index;
        reg [7:0] ch;
        begin
            token_bad = (token_len(token) != width);
            for (index = 0; index < MAX_CHARS; index = index + 1) begin
                ch = token[index*8 +: 8];
                if (ch !== 8'h00 && ch !== "0" && ch !== "1")
                    token_bad = 1'b1;
            end
        end
    endfunction


    // '0'/'1' characters to bits: character c from the right is bit c.
    function [MAX_CHARS-1:0] token_bits;
        input [MAX_CHARS*8-1:0] token;
        integer index;
        begin
            token_bits = {MAX_CHARS{1'b0}};
            for (index = 0; index < MAX_CHARS; index = index + 1)
                token_bits[index] = token[index*8];
        end
    endfunction


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
        i_scaled_uni = {`GEN_SCALED_ENTRY{1'b0}};
        i_scaled_bi = {`GEN_SCALED_ENTRY{1'b0}};
        i_unscaled = {`GEN_UNSCALED_ENTRY{1'b0}};

        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL add_gaines: observed latency 0, expected %0d", `GEN_PP_DELAY);
            $finish;
        end

        fd = $fopen("vec/add_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/add_gaines.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        // One line holds one row of exactly DATA_COLS binary columns. A line
        // carrying any other column count, a line that fills the line buffer,
        // or an x or z data field ends the run with a fatal at that row, so a
        // row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file. The multi-bit entry columns are read as
        // characters and passed to token_bad, which fatals unless the two
        // scaled columns are exactly GEN_SCALED_ENTRY characters and the
        // unscaled column exactly GEN_UNSCALED_ENTRY characters, every
        // character a 0 or a 1.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "add_gaines: row %0d fills the %0d byte line buffer", count + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(
                    line, "%s %s %s %s %s %s %s %s",
                    tok_reset_flag,
                    tok_i_scaled_uni, tok_expected_scaled_uni,
                    tok_i_scaled_bi, tok_expected_scaled_bi,
                    tok_i_unscaled, tok_expected_unscaled,
                    extra
                );
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "add_gaines: row %0d is blank", count + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "add_gaines: row %0d scanned %0d columns, expected %0d", count + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_reset_flag !== "0" && tok_reset_flag !== "1")
                        $fatal(1, "add_gaines: row %0d input field reset_flag is \"%0s\", expected a single 0 or 1", count + 1, tok_reset_flag);
                    reset_flag = (tok_reset_flag == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_expected_scaled_uni !== "0" && tok_expected_scaled_uni !== "1")
                        $fatal(1, "add_gaines: row %0d expected field expected_scaled_uni is \"%0s\", expected a single 0 or 1", count + 1, tok_expected_scaled_uni);
                    expected_scaled_uni = (tok_expected_scaled_uni == "1");
                    if (tok_expected_scaled_bi !== "0" && tok_expected_scaled_bi !== "1")
                        $fatal(1, "add_gaines: row %0d expected field expected_scaled_bi is \"%0s\", expected a single 0 or 1", count + 1, tok_expected_scaled_bi);
                    expected_scaled_bi = (tok_expected_scaled_bi == "1");
                    if (tok_expected_unscaled !== "0" && tok_expected_unscaled !== "1")
                        $fatal(1, "add_gaines: row %0d expected field expected_unscaled is \"%0s\", expected a single 0 or 1", count + 1, tok_expected_unscaled);
                    expected_unscaled = (tok_expected_unscaled == "1");
                    // Input column: a token wider than the port would be truncated into
                    // stimulus the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_i_scaled_uni, `GEN_SCALED_ENTRY))
                        $fatal(1, "add_gaines: row %0d input field i_scaled_uni is \"%0s\", expected %0d characters of 0 or 1", count + 1, tok_i_scaled_uni, `GEN_SCALED_ENTRY);
                    i_scaled_uni = token_bits(tok_i_scaled_uni);
                    // Input column: a token wider than the port would be truncated into
                    // stimulus the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_i_scaled_bi, `GEN_SCALED_ENTRY))
                        $fatal(1, "add_gaines: row %0d input field i_scaled_bi is \"%0s\", expected %0d characters of 0 or 1", count + 1, tok_i_scaled_bi, `GEN_SCALED_ENTRY);
                    i_scaled_bi = token_bits(tok_i_scaled_bi);
                    // Input column: a token wider than the port would be truncated into
                    // stimulus the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_i_unscaled, `GEN_UNSCALED_ENTRY))
                        $fatal(1, "add_gaines: row %0d input field i_unscaled is \"%0s\", expected %0d characters of 0 or 1", count + 1, tok_i_unscaled, `GEN_UNSCALED_ENTRY);
                    i_unscaled = token_bits(tok_i_unscaled);
                    if (reset_flag)
                        reset_duts;
                    #1;
                    count = count + 1;
                    if (o_scaled_uni !== expected_scaled_uni) begin
                        $display("FAIL add_gaines scaled unipolar cycle %0d", count);
                        fails = fails + 1;
                    end
                    if (o_scaled_bi !== expected_scaled_bi) begin
                        $display("FAIL add_gaines scaled bipolar cycle %0d", count);
                        fails = fails + 1;
                    end
                    if (o_unscaled !== expected_unscaled) begin
                        $display("FAIL add_gaines unscaled cycle %0d", count);
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
            $display("FAIL add_gaines: compared %0d vectors, generator wrote %0d",
                     count, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS add_gaines: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL add_gaines: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
