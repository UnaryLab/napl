`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "conv_mix/vec/conv_mix_params.vh"
// Python golden rows are <rst> <in_u> <in_b> <w_u> <w_b> <bias_u> <bias_b>
// <out_u_a> <out_u_b> <out_u_c> <out_u_d> <out_b_a> <out_b_b> <out_b_c> <out_b_d>,
// one per timestep. Polarity selects the circuit, so each row drives the unipolar
// and the bipolar DUT with its own encoded streams, in each of the four
// geometries: a (padding 1, bias), b (padding 0, no bias), c (padding 2, stride 2,
// dilation 2, bias), d (c's geometry at an explicit scale below its fan-in, the arm
// the saturation rows charge onto its clamp). Weight and bias are held fixed-point
// codes carried per row, since the RTL holds their encoders (and the bipolar pad
// encoder) and re-encodes them every timestep. Outputs are combinational in the
// arrival cycle, so each row is checked before the posedge that advances the
// sequence indices and the accumulators. rst=1 pulses i_rst_n low first.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended
// silently by %b.
// Co-sim: make test MODULE=conv_mix


module conv_mix_tb;
    localparam integer OPW      = `GEN_SEQ_WIDTH + 1;
    localparam integer IN_WIDTH = `GEN_BATCH * `GEN_IN_CHANNELS * `GEN_IN_H * `GEN_IN_W;
    localparam integer W_WIDTH  = `GEN_OUT_CHANNELS * `GEN_IN_CHANNELS
                                  * `GEN_KERNEL_H * `GEN_KERNEL_W * OPW;
    localparam integer B_WIDTH  = `GEN_OUT_CHANNELS * OPW;

    reg                    i_clk;
    reg                    i_rst_n;
    reg  [IN_WIDTH-1:0]    i_input_u;
    reg  [IN_WIDTH-1:0]    i_input_b;
    reg  [W_WIDTH-1:0]     i_weight_u;
    reg  [W_WIDTH-1:0]     i_weight_b;
    reg  [B_WIDTH-1:0]     i_bias_u;
    reg  [B_WIDTH-1:0]     i_bias_b;
    wire [`GEN_LANES_A-1:0] o_output_u_a;
    wire [`GEN_LANES_B-1:0] o_output_u_b;
    wire [`GEN_LANES_C-1:0] o_output_u_c;
    wire [`GEN_LANES_D-1:0] o_output_u_d;
    wire [`GEN_LANES_A-1:0] o_output_b_a;
    wire [`GEN_LANES_B-1:0] o_output_b_b;
    wire [`GEN_LANES_C-1:0] o_output_b_c;
    wire [`GEN_LANES_D-1:0] o_output_b_d;

    conv_mix_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_A),
        .PADDING      (`GEN_PADDING_A),
        .DILATION     (`GEN_DILATION_A),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_A),
        .HAS_BIAS     (`GEN_HAS_BIAS_A),
        .LANES        (`GEN_LANES_A)
    ) dut_u_a (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_output      (o_output_u_a)
    );

    // HAS_BIAS = 0 drops the bias addend, so entry and the default scale shrink;
    // padding 0 leaves no tap on the pad encoder.
    conv_mix_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_B),
        .PADDING      (`GEN_PADDING_B),
        .DILATION     (`GEN_DILATION_B),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_B),
        .HAS_BIAS     (`GEN_HAS_BIAS_B),
        .LANES        (`GEN_LANES_B)
    ) dut_u_b (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_output      (o_output_u_b)
    );

    conv_mix_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_C),
        .PADDING      (`GEN_PADDING_C),
        .DILATION     (`GEN_DILATION_C),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_C),
        .HAS_BIAS     (`GEN_HAS_BIAS_C),
        .LANES        (`GEN_LANES_C)
    ) dut_u_c (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_output      (o_output_u_c)
    );

    // c's geometry at an explicit scale below its fan-in: the accumulators the
    // saturation rows drive onto the clamp WIDTH sets.
    conv_mix_unipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_D),
        .PADDING      (`GEN_PADDING_D),
        .DILATION     (`GEN_DILATION_D),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_D),
        .HAS_BIAS     (`GEN_HAS_BIAS_D),
        .LANES        (`GEN_LANES_D)
    ) dut_u_d (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_output      (o_output_u_d)
    );

    conv_mix_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_A),
        .PADDING      (`GEN_PADDING_A),
        .DILATION     (`GEN_DILATION_A),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_A),
        .HAS_BIAS     (`GEN_HAS_BIAS_A),
        .LANES        (`GEN_LANES_A)
    ) dut_b_a (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_output      (o_output_b_a)
    );

    conv_mix_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_B),
        .PADDING      (`GEN_PADDING_B),
        .DILATION     (`GEN_DILATION_B),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_B),
        .HAS_BIAS     (`GEN_HAS_BIAS_B),
        .LANES        (`GEN_LANES_B)
    ) dut_b_b (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_output      (o_output_b_b)
    );

    conv_mix_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_C),
        .PADDING      (`GEN_PADDING_C),
        .DILATION     (`GEN_DILATION_C),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_C),
        .HAS_BIAS     (`GEN_HAS_BIAS_C),
        .LANES        (`GEN_LANES_C)
    ) dut_b_c (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_output      (o_output_b_c)
    );

    conv_mix_bipolar #(
        .BATCH        (`GEN_BATCH),
        .IN_CHANNELS  (`GEN_IN_CHANNELS),
        .IN_H         (`GEN_IN_H),
        .IN_W         (`GEN_IN_W),
        .OUT_CHANNELS (`GEN_OUT_CHANNELS),
        .KERNEL_H     (`GEN_KERNEL_H),
        .KERNEL_W     (`GEN_KERNEL_W),
        .STRIDE       (`GEN_STRIDE_D),
        .PADDING      (`GEN_PADDING_D),
        .DILATION     (`GEN_DILATION_D),
        .SEQ_WIDTH    (`GEN_SEQ_WIDTH),
        .WIDTH        (`GEN_WIDTH),
        .SCALE        (`GEN_SCALE_D),
        .HAS_BIAS     (`GEN_HAS_BIAS_D),
        .LANES        (`GEN_LANES_D)
    ) dut_b_d (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_output      (o_output_b_d)
    );

    // One character wider than the widest golden column: $sscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    // The widest of every column is taken here instead of assumed.
    localparam integer WIDEST_AB   = (`GEN_LANES_A > `GEN_LANES_B) ? `GEN_LANES_A : `GEN_LANES_B;
    localparam integer WIDEST_CD   = (`GEN_LANES_C > `GEN_LANES_D) ? `GEN_LANES_C : `GEN_LANES_D;
    localparam integer WIDEST_LANE = (WIDEST_AB > WIDEST_CD) ? WIDEST_AB : WIDEST_CD;
    localparam integer WIDEST_IN   = (IN_WIDTH > W_WIDTH) ? IN_WIDTH : W_WIDTH;
    localparam integer MAX_CHARS   = ((WIDEST_IN > WIDEST_LANE) ? WIDEST_IN : WIDEST_LANE) + 1;

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
    localparam integer LINE_BYTES = 2048;
    // hdr_line holds the header line alone; its size is named here because
    // the header read is guarded against it, the way rows are against LINE_BYTES.
    localparam integer HDR_BYTES  = 128;
    localparam integer DATA_COLS  = 15;
    // The field count of ROW_FMT must match DATA_COLS; the trailing %s is the
    // sentinel that catches a row carrying one column too many.
    localparam ROW_FMT = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s";

    integer fd, code, chars, n, fails;
    reg  [8*8-1:0]          extra;
    reg  [8*LINE_BYTES-1:0] line;
    reg                     rst;
    reg  [MAX_CHARS*8-1:0] tok_rst;
    reg  [8*HDR_BYTES-1:0] hdr_line;
    reg  [MAX_CHARS*8-1:0]  tok_in_u;
    reg  [MAX_CHARS*8-1:0]  tok_in_b;
    reg  [MAX_CHARS*8-1:0]  tok_w_u;
    reg  [MAX_CHARS*8-1:0]  tok_w_b;
    reg  [MAX_CHARS*8-1:0]  tok_bias_u;
    reg  [MAX_CHARS*8-1:0]  tok_bias_b;
    reg  [MAX_CHARS*8-1:0]  tok_u_a;
    reg  [MAX_CHARS*8-1:0]  tok_u_b;
    reg  [MAX_CHARS*8-1:0]  tok_u_c;
    reg  [MAX_CHARS*8-1:0]  tok_u_d;
    reg  [MAX_CHARS*8-1:0]  tok_b_a;
    reg  [MAX_CHARS*8-1:0]  tok_b_b;
    reg  [MAX_CHARS*8-1:0]  tok_b_c;
    reg  [MAX_CHARS*8-1:0]  tok_b_d;
    reg  [`GEN_LANES_A-1:0] exp_u_a;
    reg  [`GEN_LANES_B-1:0] exp_u_b;
    reg  [`GEN_LANES_C-1:0] exp_u_c;
    reg  [`GEN_LANES_D-1:0] exp_u_d;
    reg  [`GEN_LANES_A-1:0] exp_b_a;
    reg  [`GEN_LANES_B-1:0] exp_b_b;
    reg  [`GEN_LANES_C-1:0] exp_b_c;
    reg  [`GEN_LANES_D-1:0] exp_b_d;


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
                    $fatal(1, "conv_mix: row %0d %0s column %0s character %0d is %0c (0x%0h), outside the binary alphabet",
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
        i_clk           = 1'b0;
        i_rst_n         = 1'b1;
        i_input_u = {IN_WIDTH{1'b0}};
        i_input_b = {IN_WIDTH{1'b0}};
        i_weight_u      = {W_WIDTH{1'b0}};
        i_weight_b      = {W_WIDTH{1'b0}};
        i_bias_u        = {B_WIDTH{1'b0}};
        i_bias_b        = {B_WIDTH{1'b0}};

        fd = $fopen("vec/conv_mix.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/conv_mix.vec (run `make vectors` first)");
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
            $fatal(1, "conv_mix: vec file is empty, no header line");
        if (code == HDR_BYTES && hdr_line[7:0] !== "\n")
            $fatal(1, "conv_mix: header line fills the %0d byte header buffer", HDR_BYTES);

        n = 0;
        fails = 0;
        // Each row is compared before the posedge, so this testbench can only drive a
        // pp_delay of 0; that pre-posedge sampling is what enforces it. This check
        // rejects a model whose pp_delay stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL conv: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
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
                    $fatal(1, "conv_mix: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                code = $sscanf(line, ROW_FMT, tok_rst, tok_in_u, tok_in_b, tok_w_u, tok_w_b,
                                              tok_bias_u, tok_bias_b, tok_u_a, tok_u_b, tok_u_c, tok_u_d,
                                              tok_b_a, tok_b_b, tok_b_c, tok_b_d, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "conv_mix: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "conv_mix: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    check_input_column(tok_rst, 1, "rst");
                    rst = token_bits(tok_rst);
                    check_input_column(tok_in_u, IN_WIDTH, "in_u");
                    check_input_column(tok_in_b, IN_WIDTH, "in_b");
                    check_input_column(tok_w_u, W_WIDTH, "w_u");
                    check_input_column(tok_w_b, W_WIDTH, "w_b");
                    check_input_column(tok_bias_u, B_WIDTH, "bias_u");
                    check_input_column(tok_bias_b, B_WIDTH, "bias_b");
                    check_expected_column(tok_u_a, `GEN_LANES_A, "out_u_a");
                    check_expected_column(tok_u_b, `GEN_LANES_B, "out_u_b");
                    check_expected_column(tok_u_c, `GEN_LANES_C, "out_u_c");
                    check_expected_column(tok_u_d, `GEN_LANES_D, "out_u_d");
                    check_expected_column(tok_b_a, `GEN_LANES_A, "out_b_a");
                    check_expected_column(tok_b_b, `GEN_LANES_B, "out_b_b");
                    check_expected_column(tok_b_c, `GEN_LANES_C, "out_b_c");
                    check_expected_column(tok_b_d, `GEN_LANES_D, "out_b_d");

                    // reset boundary: clear every sequence index and lane accumulator
                    if (rst == 1) begin
                        i_rst_n = 1'b0;
                        #1;
                        i_rst_n = 1'b1;
                        #1;
                    end

                    i_input_u = token_bits(tok_in_u);
                    i_input_b = token_bits(tok_in_b);
                    i_weight_u      = token_bits(tok_w_u);
                    i_weight_b      = token_bits(tok_w_b);
                    i_bias_u        = token_bits(tok_bias_u);
                    i_bias_b        = token_bits(tok_bias_b);
                    exp_u_a         = token_bits(tok_u_a);
                    exp_u_b         = token_bits(tok_u_b);
                    exp_u_c         = token_bits(tok_u_c);
                    exp_u_d         = token_bits(tok_u_d);
                    exp_b_a         = token_bits(tok_b_a);
                    exp_b_b         = token_bits(tok_b_b);
                    exp_b_c         = token_bits(tok_b_c);
                    exp_b_d         = token_bits(tok_b_d);
                    #1;

                    n = n + 1;
                    if (o_output_u_a !== exp_u_a) begin
                        $display("FAIL n=%0d unipolar pad1 bias : got %b exp %b", n, o_output_u_a, exp_u_a);
                        fails = fails + 1;
                    end
                    if (o_output_u_b !== exp_u_b) begin
                        $display("FAIL n=%0d unipolar pad0 nobias : got %b exp %b", n, o_output_u_b, exp_u_b);
                        fails = fails + 1;
                    end
                    if (o_output_u_c !== exp_u_c) begin
                        $display("FAIL n=%0d unipolar strided : got %b exp %b", n, o_output_u_c, exp_u_c);
                        fails = fails + 1;
                    end
                    if (o_output_u_d !== exp_u_d) begin
                        $display("FAIL n=%0d unipolar scaled : got %b exp %b", n, o_output_u_d, exp_u_d);
                        fails = fails + 1;
                    end
                    if (o_output_b_a !== exp_b_a) begin
                        $display("FAIL n=%0d bipolar pad1 bias : got %b exp %b", n, o_output_b_a, exp_b_a);
                        fails = fails + 1;
                    end
                    if (o_output_b_b !== exp_b_b) begin
                        $display("FAIL n=%0d bipolar pad0 nobias : got %b exp %b", n, o_output_b_b, exp_b_b);
                        fails = fails + 1;
                    end
                    if (o_output_b_c !== exp_b_c) begin
                        $display("FAIL n=%0d bipolar strided : got %b exp %b", n, o_output_b_c, exp_b_c);
                        fails = fails + 1;
                    end
                    if (o_output_b_d !== exp_b_d) begin
                        $display("FAIL n=%0d bipolar scaled : got %b exp %b", n, o_output_b_d, exp_b_d);
                        fails = fails + 1;
                    end

                    // clock edge advances the sequence indices and the accumulators
                    i_clk = 1'b1; #1;
                    i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL conv: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS conv_mix: %0d/%0d vectors (%0d, %0d, %0d and %0d lanes x %0d taps, scale %0d, %0d, %0d and %0d)",
                     n, n, `GEN_LANES_A, `GEN_LANES_B, `GEN_LANES_C, `GEN_LANES_D,
                     `GEN_IN_CHANNELS * `GEN_KERNEL_H * `GEN_KERNEL_W,
                     `GEN_SCALE_A, `GEN_SCALE_B, `GEN_SCALE_C, `GEN_SCALE_D);
        else
            $display("FAIL conv: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
