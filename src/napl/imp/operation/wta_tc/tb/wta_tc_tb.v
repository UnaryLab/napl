`timescale 1ns/1ps
`default_nettype none
`include "wta_tc/vec/wta_tc_params.vh"
// Golden rows are <rst> <input> <output>, produced by the Python model, with
// `input` packing the stacked streams. The output is combinational, so it is
// checked before the posedge that advances the history and winner latch; a
// marked row resets both first.
// Co-sim: make test OP=wta_tc


module wta_tc_tb;
    reg                   i_clk;
    reg                   i_rst_n;
    reg  [`GEN_ENTRY-1:0] i_input;
    wire                  o_output;

    wta_tc #(
        .ENTRY(`GEN_ENTRY)
    ) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (i_input),
        .o_output(o_output)
    );

    // 10 ns clock period.
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
    localparam DATA_COLS  = 3;

    localparam integer MAX_CHARS = `GEN_ENTRY + 1;

    integer fd, code, chars, n, fails;
    reg rst_flag, exp_out;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_flag;
    reg [8*8-1:0] tok_exp_out;
    reg [MAX_CHARS*8-1:0] tok_i_input;
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
        fd = $fopen("vec/wta_tc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/wta_tc.vec (run `make vectors` first)");
            $finish;
        end

        i_input = {`GEN_ENTRY{1'b1}};
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // The output is a gate network over the current input and the held state.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL wta_tc: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
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
                    $fatal(1, "wta_tc: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok_rst_flag, tok_i_input, tok_exp_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "wta_tc: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "wta_tc: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_flag !== "0" && tok_rst_flag !== "1")
                        $fatal(1, "wta_tc: row %0d input field rst_flag is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_flag);
                    rst_flag = (tok_rst_flag == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "wta_tc: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    // Input column: i_input packs ENTRY stacked streams, and a token
                    // wider than the port would be truncated into
                    // stimulus the golden row never described, so its width and its
                    // characters are both checked here.
                    if (token_bad(tok_i_input, `GEN_ENTRY))
                        $fatal(1, "wta_tc: row %0d input field i_input is \"%0s\", expected %0d characters of 0 or 1", n + 1, tok_i_input, `GEN_ENTRY);
                    i_input = token_bits(tok_i_input);
                    if (rst_flag) begin
                        i_rst_n = 1'b0;
                        @(posedge i_clk);   // the async reset reloads previous and fired here
                        @(negedge i_clk);
                        i_rst_n = 1'b1;
                    end
                    #1;                     // let the combinational output settle
                    n = n + 1;
                    if (o_output !== exp_out) begin
                        $display("FAIL cycle %0d: i_input=%b got %b exp %b",
                                 n, i_input, o_output, exp_out);
                        fails = fails + 1;
                    end
                    @(posedge i_clk);       // advance the history and the winner latch
                    @(negedge i_clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL wta_tc: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS wta_tc: %0d/%0d vectors", n, n);
        else
            $display("FAIL wta_tc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
