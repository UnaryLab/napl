`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "conv_gaines/vec/conv_gaines_params.vh"
// Python golden rows are <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c>
// <out_b_a> <out_b_b> <out_b_c>, one per timestep. Polarity selects the circuit,
// so each row drives the unipolar and the bipolar DUT with its own encoded input
// stream. The unipolar arms are a (padding 1, bias, scaled), b (padding 0, no
// bias, OR adder), and c (padding 1, stride 2, dilation 2, bias, scaled); the
// bipolar arms are a and c plus b at padding 0 with a bias, since bipolar Gaines
// addition is scaled only. Weight and bias are held fixed-point codes read from
// vec/conv_gaines_operand.hex, and the threshold, bias, select and pad streams
// are built inside the DUT, so none of them is a vector column.
// Outputs are combinational in the arrival cycle, so each row is checked before
// the posedge that advances the threshold, bias, select and pad indices. rst=1
// pulses i_rst_n low first.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended silently
// by %b.
// Co-sim: make test MODULE=conv_gaines


module conv_gaines_tb;
    localparam integer OPW      = `GEN_SEQ_WIDTH + 1;
    localparam integer K        = `GEN_IN_CHANNELS * `GEN_KERNEL_H * `GEN_KERNEL_W;
    localparam integer IN_WIDTH = `GEN_BATCH * `GEN_IN_CHANNELS * `GEN_IN_H * `GEN_IN_W;
    localparam integer W_WIDTH  = `GEN_OUT_CHANNELS * K * OPW;
    localparam integer B_WIDTH  = `GEN_OUT_CHANNELS * OPW;
    localparam integer OP_COUNT = `GEN_OUT_CHANNELS * K + `GEN_OUT_CHANNELS;

    reg                       i_clk;
    reg                       i_rst_n;
    reg  [IN_WIDTH-1:0]       i_input_u;
    reg  [IN_WIDTH-1:0]       i_input_b;
    wire [`GEN_LANES_U_A-1:0] o_output_u_a;
    wire [`GEN_LANES_U_B-1:0] o_output_u_b;
    wire [`GEN_LANES_U_C-1:0] o_output_u_c;
    wire [`GEN_LANES_B_A-1:0] o_output_b_a;
    wire [`GEN_LANES_B_B-1:0] o_output_b_b;
    wire [`GEN_LANES_B_C-1:0] o_output_b_c;

    // Held fixed-point operands: OUT_CHANNELS*K weight codes, then OUT_CHANNELS
    // bias codes. Generated from the model, so the DUT sees its exact operands.
    // A code is the probability both polarities compare against, so one table
    // drives every DUT.
    reg [OPW-1:0] operand [0:OP_COUNT-1];
    initial $readmemb("vec/conv_gaines_operand.hex", operand);

    wire [W_WIDTH-1:0] i_weight;
    wire [B_WIDTH-1:0] i_bias;

    genvar channel, tap;
    generate
        for (channel = 0; channel < `GEN_OUT_CHANNELS; channel = channel + 1) begin : g_operand
            for (tap = 0; tap < K; tap = tap + 1) begin : g_weight
                assign i_weight[(channel*K + tap)*OPW +: OPW] = operand[channel*K + tap];
            end
            assign i_bias[channel*OPW +: OPW] =
                operand[`GEN_OUT_CHANNELS * K + channel];
        end
    endgenerate

    conv_gaines_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_U_A),
        .PADDING      (`GEN_PADDING_U_A),
        .DILATION     (`GEN_DILATION_U_A),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH  (`GEN_SCALE_WIDTH),
        .HAS_BIAS     (`GEN_HAS_BIAS_U_A),
        .SCALED       (`GEN_SCALED_U_A),
        .LANES        (`GEN_LANES_U_A)
    ) dut_u_a (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input_u),
        .i_weight (i_weight),
        .i_bias   (i_bias),
        .o_output (o_output_u_a)
    );

    // HAS_BIAS = 0 drops the bias addend and SCALED = 0 selects the OR adder, so
    // the scaled path is not the only Gaines adder the co-simulation covers;
    // padding 0 leaves no tap on the pad path.
    conv_gaines_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_U_B),
        .PADDING      (`GEN_PADDING_U_B),
        .DILATION     (`GEN_DILATION_U_B),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH  (`GEN_SCALE_WIDTH),
        .HAS_BIAS     (`GEN_HAS_BIAS_U_B),
        .SCALED       (`GEN_SCALED_U_B),
        .LANES        (`GEN_LANES_U_B)
    ) dut_u_b (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input_u),
        .i_weight (i_weight),
        .i_bias   (i_bias),
        .o_output (o_output_u_b)
    );

    conv_gaines_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_U_C),
        .PADDING      (`GEN_PADDING_U_C),
        .DILATION     (`GEN_DILATION_U_C),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH  (`GEN_SCALE_WIDTH),
        .HAS_BIAS     (`GEN_HAS_BIAS_U_C),
        .SCALED       (`GEN_SCALED_U_C),
        .LANES        (`GEN_LANES_U_C)
    ) dut_u_c (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input_u),
        .i_weight (i_weight),
        .i_bias   (i_bias),
        .o_output (o_output_u_c)
    );

    conv_gaines_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_B_A),
        .PADDING      (`GEN_PADDING_B_A),
        .DILATION     (`GEN_DILATION_B_A),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH  (`GEN_SCALE_WIDTH),
        .HAS_BIAS     (`GEN_HAS_BIAS_B_A),
        .LANES        (`GEN_LANES_B_A)
    ) dut_b_a (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input_b),
        .i_weight (i_weight),
        .i_bias   (i_bias),
        .o_output (o_output_b_a)
    );

    // Padding 0 puts no tap on the pad stream, so the bipolar pad encoder is
    // covered by a and c and its absence by b.
    conv_gaines_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_B_B),
        .PADDING      (`GEN_PADDING_B_B),
        .DILATION     (`GEN_DILATION_B_B),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH  (`GEN_SCALE_WIDTH),
        .HAS_BIAS     (`GEN_HAS_BIAS_B_B),
        .LANES        (`GEN_LANES_B_B)
    ) dut_b_b (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input_b),
        .i_weight (i_weight),
        .i_bias   (i_bias),
        .o_output (o_output_b_b)
    );

    conv_gaines_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_B_C),
        .PADDING      (`GEN_PADDING_B_C),
        .DILATION     (`GEN_DILATION_B_C),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH  (`GEN_SCALE_WIDTH),
        .HAS_BIAS     (`GEN_HAS_BIAS_B_C),
        .LANES        (`GEN_LANES_B_C)
    ) dut_b_c (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input_b),
        .i_weight (i_weight),
        .i_bias   (i_bias),
        .o_output (o_output_b_c)
    );

    // One character wider than the widest golden column: $sscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it
    // should be saturates at the expected length and an over-long column would
    // pass the width check. The spare character makes an over-long column read
    // back long. The widest of every column is taken here instead of assumed.
    localparam integer WIDEST_U   = (`GEN_LANES_U_A > `GEN_LANES_U_B)
                                    ? ((`GEN_LANES_U_A > `GEN_LANES_U_C) ? `GEN_LANES_U_A : `GEN_LANES_U_C)
                                    : ((`GEN_LANES_U_B > `GEN_LANES_U_C) ? `GEN_LANES_U_B : `GEN_LANES_U_C);
    localparam integer WIDEST_B   = (`GEN_LANES_B_A > `GEN_LANES_B_B)
                                    ? ((`GEN_LANES_B_A > `GEN_LANES_B_C) ? `GEN_LANES_B_A : `GEN_LANES_B_C)
                                    : ((`GEN_LANES_B_B > `GEN_LANES_B_C) ? `GEN_LANES_B_B : `GEN_LANES_B_C);
    localparam integer WIDEST_LANE = (WIDEST_U > WIDEST_B) ? WIDEST_U : WIDEST_B;
    localparam integer MAX_CHARS   = ((IN_WIDTH > WIDEST_LANE) ? IN_WIDTH : WIDEST_LANE) + 1;

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
    localparam integer LINE_BYTES = 320;
    // hdr_line holds the header line alone; its size is named here because
    // the header read is guarded against it, the way rows are against LINE_BYTES.
    localparam integer HDR_BYTES  = 128;
    localparam integer DATA_COLS  = 9;
    // The field count of ROW_FMT must match DATA_COLS; the trailing %s is the
    // sentinel that catches a row carrying one column too many.
    localparam ROW_FMT = "%s %s %s %s %s %s %s %s %s %s";

    integer fd, code, chars, n, fails;
    reg  [8*8-1:0]          extra;
    reg  [8*LINE_BYTES-1:0] line;
    reg                       rst;
    reg  [MAX_CHARS*8-1:0] tok_rst;
    reg  [8*HDR_BYTES-1:0] hdr_line;
    reg  [MAX_CHARS*8-1:0]    tok_in_u;
    reg  [MAX_CHARS*8-1:0]    tok_in_b;
    reg  [MAX_CHARS*8-1:0]    tok_u_a;
    reg  [MAX_CHARS*8-1:0]    tok_u_b;
    reg  [MAX_CHARS*8-1:0]    tok_u_c;
    reg  [MAX_CHARS*8-1:0]    tok_b_a;
    reg  [MAX_CHARS*8-1:0]    tok_b_b;
    reg  [MAX_CHARS*8-1:0]    tok_b_c;
    reg  [`GEN_LANES_U_A-1:0] exp_u_a;
    reg  [`GEN_LANES_U_B-1:0] exp_u_b;
    reg  [`GEN_LANES_U_C-1:0] exp_u_c;
    reg  [`GEN_LANES_B_A-1:0] exp_b_a;
    reg  [`GEN_LANES_B_B-1:0] exp_b_b;
    reg  [`GEN_LANES_B_C-1:0] exp_b_c;


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
                    $fatal(1, "conv_gaines: row %0d %0s column %0s character %0d is %0c (0x%0h), outside the binary alphabet",
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
        i_clk     = 1'b0;
        i_rst_n   = 1'b1;
        i_input_u = {IN_WIDTH{1'b0}};
        i_input_b = {IN_WIDTH{1'b0}};

        fd = $fopen("vec/conv_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/conv_gaines.vec (run `make vectors` first)");
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
            $fatal(1, "conv_gaines: vec file is empty, no header line");
        if (code == HDR_BYTES && hdr_line[7:0] !== "\n")
            $fatal(1, "conv_gaines: header line fills the %0d byte header buffer", HDR_BYTES);

        n = 0;
        fails = 0;
        // Each row is compared before the posedge, so this testbench can only drive a
        // pp_delay of 0; that pre-posedge sampling is what enforces it. This check
        // rejects a model whose pp_delay stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL conv_gaines: testbench samples combinationally, model pp_delay is %0d",
                     `GEN_PP_DELAY);
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
                    $fatal(1, "conv_gaines: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                code = $sscanf(line, ROW_FMT, tok_rst, tok_in_u, tok_in_b, tok_u_a, tok_u_b,
                                              tok_u_c, tok_b_a, tok_b_b, tok_b_c, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "conv_gaines: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "conv_gaines: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    check_input_column(tok_rst, 1, "rst");
                    rst = token_bits(tok_rst);
                    check_input_column(tok_in_u, IN_WIDTH, "in_u");
                    check_input_column(tok_in_b, IN_WIDTH, "in_b");
                    check_expected_column(tok_u_a, `GEN_LANES_U_A, "out_u_a");
                    check_expected_column(tok_u_b, `GEN_LANES_U_B, "out_u_b");
                    check_expected_column(tok_u_c, `GEN_LANES_U_C, "out_u_c");
                    check_expected_column(tok_b_a, `GEN_LANES_B_A, "out_b_a");
                    check_expected_column(tok_b_b, `GEN_LANES_B_B, "out_b_b");
                    check_expected_column(tok_b_c, `GEN_LANES_B_C, "out_b_c");

                    // reset boundary: clear every sequence index
                    if (rst == 1) begin
                        i_rst_n = 1'b0;
                        #1;
                        i_rst_n = 1'b1;
                        #1;
                    end

                    i_input_u = token_bits(tok_in_u);
                    i_input_b = token_bits(tok_in_b);
                    exp_u_a   = token_bits(tok_u_a);
                    exp_u_b   = token_bits(tok_u_b);
                    exp_u_c   = token_bits(tok_u_c);
                    exp_b_a   = token_bits(tok_b_a);
                    exp_b_b   = token_bits(tok_b_b);
                    exp_b_c   = token_bits(tok_b_c);
                    #1;

                    n = n + 1;
                    if (o_output_u_a !== exp_u_a) begin
                        $display("FAIL n=%0d unipolar pad1 bias scaled : got %b exp %b", n, o_output_u_a, exp_u_a);
                        fails = fails + 1;
                    end
                    if (o_output_u_b !== exp_u_b) begin
                        $display("FAIL n=%0d unipolar pad0 nobias or : got %b exp %b", n, o_output_u_b, exp_u_b);
                        fails = fails + 1;
                    end
                    if (o_output_u_c !== exp_u_c) begin
                        $display("FAIL n=%0d unipolar strided : got %b exp %b", n, o_output_u_c, exp_u_c);
                        fails = fails + 1;
                    end
                    if (o_output_b_a !== exp_b_a) begin
                        $display("FAIL n=%0d bipolar pad1 bias : got %b exp %b", n, o_output_b_a, exp_b_a);
                        fails = fails + 1;
                    end
                    if (o_output_b_b !== exp_b_b) begin
                        $display("FAIL n=%0d bipolar pad0 bias : got %b exp %b", n, o_output_b_b, exp_b_b);
                        fails = fails + 1;
                    end
                    if (o_output_b_c !== exp_b_c) begin
                        $display("FAIL n=%0d bipolar strided : got %b exp %b", n, o_output_b_c, exp_b_c);
                        fails = fails + 1;
                    end

                    // clock edge advances the sequence indices
                    i_clk = 1'b1; #1;
                    i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL conv_gaines: consumed %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS conv_gaines: %0d/%0d vectors (%0d, %0d and %0d unipolar lanes, %0d, %0d and %0d bipolar lanes x %0d taps, scale width %0d)",
                     n, n, `GEN_LANES_U_A, `GEN_LANES_U_B, `GEN_LANES_U_C,
                     `GEN_LANES_B_A, `GEN_LANES_B_B, `GEN_LANES_B_C, K, `GEN_SCALE_WIDTH);
        else
            $display("FAIL conv_gaines: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
