`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "add_scale/vec/add_scale_params.vh"
// Golden rows are <rst> <unipolar input/output> <bipolar input/output> followed by the
// same pair for the narrow instances (ENTRY_N <= SCALE_N) and for the boundary
// instances (ENTRY_B = SCALE_B + 1). Reset precedes the marked row; outputs are
// checked before the posedge updates state.
// Co-sim: make test OP=add_scale


module add_scale_tb;
    reg              i_clk;
    reg              i_rst_n;
    reg  [`GEN_ENTRY-1:0] i_input_uni;
    reg  [`GEN_ENTRY-1:0] i_input_bi;
    reg  [`GEN_ENTRY_N-1:0] i_input_uni_n;
    reg  [`GEN_ENTRY_N-1:0] i_input_bi_n;
    reg  [`GEN_ENTRY_B-1:0] i_input_uni_b;
    reg  [`GEN_ENTRY_B-1:0] i_input_bi_b;
    wire             o_uni;
    wire             o_bi;
    wire             o_uni_n;
    wire             o_bi_n;
    wire             o_uni_b;
    wire             o_bi_b;

    add_scale_unipolar #(
        .SCALE(`GEN_SCALE),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY)
    ) dut_uni (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni),
        .o_output  (o_uni)
    );

    add_scale_bipolar #(
        .SCALE(`GEN_SCALE),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY)
    ) dut_bi (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi),
        .o_output  (o_bi)
    );

    add_scale_unipolar #(
        .SCALE(`GEN_SCALE_N),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY_N)
    ) dut_uni_n (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni_n),
        .o_output  (o_uni_n)
    );

    add_scale_bipolar #(
        .SCALE(`GEN_SCALE_N),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY_N)
    ) dut_bi_n (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi_n),
        .o_output  (o_bi_n)
    );

    add_scale_unipolar #(
        .SCALE(`GEN_SCALE_B),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY_B)
    ) dut_uni_b (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni_b),
        .o_output  (o_uni_b)
    );

    add_scale_bipolar #(
        .SCALE(`GEN_SCALE_B),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY_B)
    ) dut_bi_b (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi_b),
        .o_output  (o_bi_b)
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
    localparam LINE_BYTES = 64;
    localparam DATA_COLS  = 13;

    localparam integer MAX_WN = (`GEN_ENTRY > `GEN_ENTRY_N) ? `GEN_ENTRY : `GEN_ENTRY_N;
    localparam integer MAX_ENTRY = (MAX_WN > `GEN_ENTRY_B) ? MAX_WN : `GEN_ENTRY_B;
    localparam integer MAX_CHARS = MAX_ENTRY + 1;

    integer fd, code, chars, n, fails;
    reg rst_flag, exp_uni, exp_bi, exp_uni_n, exp_bi_n, exp_uni_b, exp_bi_b;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_flag;
    reg [8*8-1:0] tok_exp_uni;
    reg [8*8-1:0] tok_exp_bi;
    reg [8*8-1:0] tok_exp_uni_n;
    reg [8*8-1:0] tok_exp_bi_n;
    reg [8*8-1:0] tok_exp_uni_b;
    reg [8*8-1:0] tok_exp_bi_b;
    reg [MAX_CHARS*8-1:0] tok_i_input_uni;
    reg [MAX_CHARS*8-1:0] tok_i_input_bi;
    reg [MAX_CHARS*8-1:0] tok_i_input_uni_n;
    reg [MAX_CHARS*8-1:0] tok_i_input_bi_n;
    reg [MAX_CHARS*8-1:0] tok_i_input_uni_b;
    reg [MAX_CHARS*8-1:0] tok_i_input_bi_b;
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


    initial begin
        fd = $fopen("vec/add_scale.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/add_scale.vec (run `make vectors` first)");
            $finish;
        end

        i_input_uni = {`GEN_ENTRY{1'b0}};
        i_input_bi = {`GEN_ENTRY{1'b0}};
        i_input_uni_n = {`GEN_ENTRY_N{1'b0}};
        i_input_bi_n = {`GEN_ENTRY_N{1'b0}};
        i_input_uni_b = {`GEN_ENTRY_B{1'b0}};
        i_input_bi_b = {`GEN_ENTRY_B{1'b0}};
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // One line holds one row of exactly DATA_COLS binary columns. A line
        // carrying any other column count, a line that fills the line buffer,
        // or an x or z data field ends the run with a fatal at that row, so a
        // row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file. The multi-bit entry columns are read as
        // characters and passed to token_bad, which fatals unless each is
        // exactly GEN_ENTRY (GEN_ENTRY_N for the narrow pair, GEN_ENTRY_B for the
        // boundary pair) characters, every character a 0 or a 1.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "add_scale: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                               tok_rst_flag, tok_i_input_uni, tok_exp_uni, tok_i_input_bi, tok_exp_bi,
                               tok_i_input_uni_n, tok_exp_uni_n, tok_i_input_bi_n, tok_exp_bi_n,
                               tok_i_input_uni_b, tok_exp_uni_b, tok_i_input_bi_b, tok_exp_bi_b, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "add_scale: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "add_scale: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_flag !== "0" && tok_rst_flag !== "1")
                        $fatal(1, "add_scale: row %0d input field rst_flag is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_flag);
                    rst_flag = (tok_rst_flag == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_uni !== "0" && tok_exp_uni !== "1")
                        $fatal(1, "add_scale: row %0d expected field exp_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni);
                    exp_uni = (tok_exp_uni == "1");
                    if (tok_exp_bi !== "0" && tok_exp_bi !== "1")
                        $fatal(1, "add_scale: row %0d expected field exp_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi);
                    exp_bi = (tok_exp_bi == "1");
                    if (tok_exp_uni_n !== "0" && tok_exp_uni_n !== "1")
                        $fatal(1, "add_scale: row %0d expected field exp_uni_n is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni_n);
                    exp_uni_n = (tok_exp_uni_n == "1");
                    if (tok_exp_bi_n !== "0" && tok_exp_bi_n !== "1")
                        $fatal(1, "add_scale: row %0d expected field exp_bi_n is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi_n);
                    exp_bi_n = (tok_exp_bi_n == "1");
                    if (tok_exp_uni_b !== "0" && tok_exp_uni_b !== "1")
                        $fatal(1, "add_scale: row %0d expected field exp_uni_b is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni_b);
                    exp_uni_b = (tok_exp_uni_b == "1");
                    if (tok_exp_bi_b !== "0" && tok_exp_bi_b !== "1")
                        $fatal(1, "add_scale: row %0d expected field exp_bi_b is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi_b);
                    exp_bi_b = (tok_exp_bi_b == "1");
                    // Input column: a token wider than the port would be truncated into
                    // stimulus the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_i_input_uni, `GEN_ENTRY))
                        $fatal(1, "add_scale: row %0d input field i_input_uni is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input_uni, `GEN_ENTRY);
                    i_input_uni = token_bits(tok_i_input_uni);
                    // Input column: a token wider than the port would be truncated into
                    // stimulus the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_i_input_bi, `GEN_ENTRY))
                        $fatal(1, "add_scale: row %0d input field i_input_bi is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input_bi, `GEN_ENTRY);
                    i_input_bi = token_bits(tok_i_input_bi);
                    if (token_bad(tok_i_input_uni_n, `GEN_ENTRY_N))
                        $fatal(1, "add_scale: row %0d input field i_input_uni_n is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input_uni_n, `GEN_ENTRY_N);
                    i_input_uni_n = token_bits(tok_i_input_uni_n);
                    if (token_bad(tok_i_input_bi_n, `GEN_ENTRY_N))
                        $fatal(1, "add_scale: row %0d input field i_input_bi_n is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input_bi_n, `GEN_ENTRY_N);
                    i_input_bi_n = token_bits(tok_i_input_bi_n);
                    if (token_bad(tok_i_input_uni_b, `GEN_ENTRY_B))
                        $fatal(1, "add_scale: row %0d input field i_input_uni_b is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input_uni_b, `GEN_ENTRY_B);
                    i_input_uni_b = token_bits(tok_i_input_uni_b);
                    if (token_bad(tok_i_input_bi_b, `GEN_ENTRY_B))
                        $fatal(1, "add_scale: row %0d input field i_input_bi_b is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input_bi_b, `GEN_ENTRY_B);
                    i_input_bi_b = token_bits(tok_i_input_bi_b);
                    if (rst_flag) begin
                        i_rst_n = 1'b0;
                        @(posedge i_clk);   // async reset fires here (acc <= 0)
                        @(negedge i_clk);
                        i_rst_n = 1'b1;
                    end
                    #1;                     // let the combinational outputs settle
                    n = n + 1;
                    if (o_uni !== exp_uni) begin
                        $display("FAIL cycle %0d unipolar: i_input=%b got %b exp %b", n, i_input_uni, o_uni, exp_uni);
                        fails = fails + 1;
                    end
                    if (o_bi !== exp_bi) begin
                        $display("FAIL cycle %0d bipolar: i_input=%b got %b exp %b", n, i_input_bi, o_bi, exp_bi);
                        fails = fails + 1;
                    end
                    if (o_uni_n !== exp_uni_n) begin
                        $display("FAIL cycle %0d narrow unipolar: i_input=%b got %b exp %b", n, i_input_uni_n, o_uni_n, exp_uni_n);
                        fails = fails + 1;
                    end
                    if (o_bi_n !== exp_bi_n) begin
                        $display("FAIL cycle %0d narrow bipolar: i_input=%b got %b exp %b", n, i_input_bi_n, o_bi_n, exp_bi_n);
                        fails = fails + 1;
                    end
                    if (o_uni_b !== exp_uni_b) begin
                        $display("FAIL cycle %0d boundary unipolar: i_input=%b got %b exp %b", n, i_input_uni_b, o_uni_b, exp_uni_b);
                        fails = fails + 1;
                    end
                    if (o_bi_b !== exp_bi_b) begin
                        $display("FAIL cycle %0d boundary bipolar: i_input=%b got %b exp %b", n, i_input_bi_b, o_bi_b, exp_bi_b);
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
            $display("FAIL add_scale: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS add_scale: %0d/%0d vectors (unipolar + bipolar, wide, narrow and boundary)", n, n);
        else
            $display("FAIL add_scale: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
