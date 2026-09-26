`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "div_scale_dyn/vec/div_scale_dyn_params.vh"
// Golden rows are <rst> <scale> <unipolar input/output> <bipolar input/output>. Reset
// precedes the marked row; outputs are checked before the posedge updates state.
// The scale column is scanned as characters and checked for SCALE_W characters of
// 0 or 1, because %b would zero-extend a short column silently and pass at the
// wrong width.
// Co-sim: make test OP=div_scale_dyn


module div_scale_dyn_tb;
    localparam integer MAX_CHARS = `GEN_SCALE_W + 1;

    reg                    i_clk;
    reg                    i_rst_n;
    reg                    i_input_uni;
    reg                    i_input_bi;
    reg [`GEN_SCALE_W-1:0] i_scale;
    wire                   o_uni;
    wire                   o_bi;

    div_scale_dyn_unipolar #(
        .SCALE_W(`GEN_SCALE_W)
    ) dut_uni (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni),
        .i_scale   (i_scale),
        .o_output  (o_uni)
    );

    div_scale_dyn_bipolar #(
        .SCALE_W(`GEN_SCALE_W),
        .WIDTH(`GEN_WIDTH)
    ) dut_bi (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi),
        .i_scale   (i_scale),
        .o_output  (o_bi)
    );

    // 10ns clock
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

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
    localparam DATA_COLS  = 6;

    integer fd, code, chars, n, fails;
    reg rst_flag, exp_uni, exp_bi;
    reg [MAX_CHARS*8-1:0] tok_scale;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_flag;
    reg [8*8-1:0] tok_i_input_uni;
    reg [8*8-1:0] tok_exp_uni;
    reg [8*8-1:0] tok_i_input_bi;
    reg [8*8-1:0] tok_exp_bi;
    reg [8*LINE_BYTES-1:0] line;

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


    initial begin
        fd = $fopen("vec/div_scale_dyn.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/div_scale_dyn.vec (run `make vectors` first)");
            $finish;
        end

        i_input_uni = 1'b0;
        i_input_bi = 1'b0;
        i_scale = {`GEN_SCALE_W{1'b0}};
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // One line holds one row of exactly DATA_COLS columns: the scale column is
        // scanned as characters, the rest as binary. A line carrying any other
        // column count, a line that fills the line buffer, or an x or z data field
        // ends the run with a fatal at that row, so a row can never borrow a
        // column from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "div_scale_dyn: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(
                    line, "%s %s %s %s %s %s %s",
                    tok_rst_flag, tok_scale, tok_i_input_uni, tok_exp_uni, tok_i_input_bi, tok_exp_bi, extra
                );
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "div_scale_dyn: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "div_scale_dyn: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_flag !== "0" && tok_rst_flag !== "1")
                        $fatal(1, "div_scale_dyn: row %0d input field rst_flag is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_flag);
                    rst_flag = (tok_rst_flag == "1");
                    if (tok_i_input_uni !== "0" && tok_i_input_uni !== "1")
                        $fatal(1, "div_scale_dyn: row %0d input field i_input_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_i_input_uni);
                    i_input_uni = (tok_i_input_uni == "1");
                    if (tok_i_input_bi !== "0" && tok_i_input_bi !== "1")
                        $fatal(1, "div_scale_dyn: row %0d input field i_input_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_i_input_bi);
                    i_input_bi = (tok_i_input_bi == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_uni !== "0" && tok_exp_uni !== "1")
                        $fatal(1, "div_scale_dyn: row %0d expected field exp_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni);
                    exp_uni = (tok_exp_uni == "1");
                    if (tok_exp_bi !== "0" && tok_exp_bi !== "1")
                        $fatal(1, "div_scale_dyn: row %0d expected field exp_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi);
                    exp_bi = (tok_exp_bi == "1");
                    // Input column: a token of the wrong width, or one carrying a
                    // character other than 0 or 1, would be converted into stimulus
                    // the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_scale, `GEN_SCALE_W))
                        $fatal(1, "div_scale_dyn: row %0d input field i_scale is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_scale, `GEN_SCALE_W);
                    i_scale = token_bits(tok_scale);
                    if (rst_flag) begin
                        i_rst_n = 1'b0;
                        @(posedge i_clk);   // async reset fires here (acc <= 0)
                        @(negedge i_clk);
                        i_rst_n = 1'b1;
                    end
                    #1;                     // let the combinational outputs settle
                    n = n + 1;
                    if (o_uni !== exp_uni) begin
                        $display("FAIL cycle %0d unipolar: i_input=%b i_scale=%b got %b exp %b",
                                 n, i_input_uni, i_scale, o_uni, exp_uni);
                        fails = fails + 1;
                    end
                    if (o_bi !== exp_bi) begin
                        $display("FAIL cycle %0d bipolar: i_input=%b i_scale=%b got %b exp %b",
                                 n, i_input_bi, i_scale, o_bi, exp_bi);
                        fails = fails + 1;
                    end
                    @(posedge i_clk);       // advance the accumulator state
                    @(negedge i_clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL div_scale_dyn: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS div_scale_dyn: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL div_scale_dyn: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
