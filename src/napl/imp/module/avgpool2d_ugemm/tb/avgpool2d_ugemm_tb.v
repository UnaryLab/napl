`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "avgpool2d_ugemm/vec/avgpool2d_ugemm_params.vh"
// Python golden rows are <rst> <in_bits> <out_bits>, one per timestep.
// Each vector column is scanned as text and its character count is checked against
// the port width before it is converted to bits, so a row that is not exactly as
// wide as its port fails here instead of being zero-extended silently by %b.
// Outputs are combinational in the arrival cycle, so each row is checked before the
// posedge that advances the lane accumulators. rst=1 pulses i_rst_n low first.
// Co-sim: make test MODULE=avgpool2d_ugemm


module avgpool2d_ugemm_tb;
    localparam integer IN_WIDTH = `GEN_LANES * `GEN_KERNEL_AREA;

    reg                 i_clk;
    reg                 i_rst_n;
    reg  [IN_WIDTH-1:0] i_input;
    wire [`GEN_LANES-1:0] o_output;

    avgpool2d_ugemm #(
        .KERNEL_AREA (`GEN_KERNEL_AREA),
        .LANES       (`GEN_LANES)
    ) dut (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input),
        .o_output      (o_output)
    );

    // One character wider than the widest golden column: $sscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    localparam integer MAX_CHARS = IN_WIDTH + 1;

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
    localparam integer LINE_BYTES = 1280;
    // hdr_line holds the header line alone; its size is named here because
    // the header read is guarded against it, the way rows are against LINE_BYTES.
    localparam integer HDR_BYTES  = 128;
    localparam integer DATA_COLS  = 3;
    // The field count of ROW_FMT must match DATA_COLS; the trailing %s is the
    // sentinel that catches a row carrying one column too many.
    localparam ROW_FMT = "%s %s %s %s";

    integer fd, code, chars, n, fails;
    reg  [8*8-1:0]          extra;
    reg  [8*LINE_BYTES-1:0] line;
    reg                   rst;
    reg  [MAX_CHARS*8-1:0] tok_rst;
    reg  [IN_WIDTH-1:0]   in_bits;
    reg  [`GEN_LANES-1:0] exp_out;
    reg  [8*HDR_BYTES-1:0] hdr_line;
    reg  [MAX_CHARS*8-1:0] tok_in;
    reg  [MAX_CHARS*8-1:0] tok_out;


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


    // A short column would be zero-extended by %b and pass, so reject it here.
    task check_width;
        input [MAX_CHARS*8-1:0] token;
        input integer expected;
        input [127:0] label;
        begin
            if (token_len(token) != expected) begin
                $display("FAIL n=%0d %0s: golden column is %0d bits, port is %0d",
                         n, label, token_len(token), expected);
                fails = fails + 1;
            end
        end
    endtask


    // token_bits takes bit 0 of each character, so a character outside the binary
    // alphabet converts silently: '2' is 8'h32 and reads back as 0. Every
    // character of a golden token must be '0' or '1'.
    task check_alphabet;
        input [MAX_CHARS*8-1:0] token;
        input [127:0]           label;
        input [127:0]           kind;
        integer index, len;
        reg [7:0] ch;
        begin
            len = token_len(token);
            for (index = 0; index < len; index = index + 1) begin
                ch = token[(len - 1 - index)*8 +: 8];
                if (ch !== "0" && ch !== "1")
                    $fatal(1, "avgpool2d_ugemm: row %0d %0s column %0s character %0d is %0c (0x%0h), outside the binary alphabet",
                           n + 1, kind, label, index, ch, ch);
            end
        end
    endtask

    // Expected column. A corrupt character here can convert to the same bits as
    // the character it replaced, so a wrong DUT output compares equal.
    task check_expected_column;
        input [MAX_CHARS*8-1:0] token;
        input integer           expected;
        input [127:0]           label;
        begin
            check_width(token, expected, label);
            check_alphabet(token, label, "expected");
        end
    endtask

    // Input column. A corrupt character here means the stimulus is not the one
    // the generator wrote, so the row proves nothing about the golden answer.
    task check_input_column;
        input [MAX_CHARS*8-1:0] token;
        input integer           expected;
        input [127:0]           label;
        begin
            check_width(token, expected, label);
            check_alphabet(token, label, "input");
        end
    endtask

    initial begin
        i_clk         = 1'b0;
        i_rst_n       = 1'b1;
        i_input = {IN_WIDTH{1'b0}};

        fd = $fopen("vec/avgpool2d_ugemm.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/avgpool2d_ugemm.vec (run `make vectors` first)");
            $finish;
        end

        // Header guard, separate from the column checks below: the header read is
        // the one read the row loop's structure guards never cover. A return of
        // zero means the file carries no header line at all, and a header that
        // fills the buffer leaves its tail unread for the row loop to scan as if
        // it were data. The vector count guard is not a backstop for either case:
        // a padded header whose swallowed bytes carry a wrong-valued row still
        // lands the count on the expected total whenever a whole row fits inside
        // the header buffer, which is a function of row width alone.
        code = $fgets(hdr_line, fd);
        if (code == 0)
            $fatal(1, "avgpool2d_ugemm: vec file is empty, no header line");
        if (code == HDR_BYTES && hdr_line[7:0] !== "\n")
            $fatal(1, "avgpool2d_ugemm: header line fills the %0d byte header buffer", HDR_BYTES);

        n = 0;
        fails = 0;
        // Each row is compared before the posedge, so this testbench can only drive a
        // pp_delay of 0; that pre-posedge sampling is what enforces it. This check
        // rejects a model whose pp_delay stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL avgpool2d_ugemm: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // One line holds one row of exactly DATA_COLS columns. A line carrying any
        // other column count, a line that fills the line buffer, or an unknown rst
        // field ends the run with a fatal at that row, so a row can never borrow a
        // column from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "avgpool2d_ugemm: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                code = $sscanf(line, ROW_FMT, tok_rst, tok_in, tok_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "avgpool2d_ugemm: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "avgpool2d_ugemm: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    check_input_column(tok_rst, 1, "rst");
                    rst = token_bits(tok_rst);
                    check_input_column(tok_in, IN_WIDTH, "in_bits");
                    check_expected_column(tok_out, `GEN_LANES, "out_bits");
                    in_bits = token_bits(tok_in);
                    exp_out = token_bits(tok_out);

                    // reset boundary: clear every lane accumulator before this timestep
                    if (rst == 1) begin
                        i_rst_n = 1'b0;
                        #1;
                        i_rst_n = 1'b1;
                        #1;
                    end

                    i_input = in_bits;
                    #1;

                    n = n + 1;
                    if (o_output !== exp_out) begin
                        $display("FAIL n=%0d : got %b exp %b", n, o_output, exp_out);
                        fails = fails + 1;
                    end

                    // clock edge advances the lane accumulators for the next timestep
                    i_clk = 1'b1; #1;
                    i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL avgpool2d_ugemm: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS avgpool2d_ugemm: %0d/%0d vectors (%0d lanes, window area %0d)",
                     n, n, `GEN_LANES, `GEN_KERNEL_AREA);
        else
            $display("FAIL avgpool2d_ugemm: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
