`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "conv_ugemm/vec/conv_ugemm_params.vh"
// Python golden rows are <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c>
// <out_u_d> <out_b_a> <out_b_b> <out_b_c> <out_b_d>, one per timestep. Polarity
// selects the circuit, so each row drives the unipolar and the bipolar DUT with
// its own encoded input stream, in each of the four geometries: a (padding 1,
// bias), b (padding 0, no bias), c (padding 1, stride 2, dilation 2, bias), and
// d (b's geometry with a bias, at an explicit scale below its fan-in). Weight and
// bias are held fixed-point codes read from vec/conv_ugemm_operand.hex, geometry
// d's railed codes from vec/conv_ugemm_operand_d.hex, and the pad stream is
// built inside the DUT, so none of them is a vector column.
// Outputs are combinational in the arrival cycle, so each row is checked before
// the posedge that advances the sequence indices, the bias and pad counters, and
// the accumulators. rst=1 pulses i_rst_n low first.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended
// silently by %b.
// Co-sim: make test MODULE=conv_ugemm


module conv_ugemm_tb;
    localparam integer OPW      = `GEN_SEQ_WIDTH + 1;
    localparam integer K        = `GEN_IN_CHANNELS * `GEN_KERNEL_H * `GEN_KERNEL_W;
    localparam integer IN_WIDTH = `GEN_BATCH * `GEN_IN_CHANNELS * `GEN_IN_H * `GEN_IN_W;
    localparam integer W_WIDTH  = `GEN_OUT_CHANNELS * K * OPW;
    localparam integer B_WIDTH  = `GEN_OUT_CHANNELS * OPW;
    localparam integer OP_COUNT = `GEN_OUT_CHANNELS * K + `GEN_OUT_CHANNELS;

    reg                     i_clk;
    reg                     i_rst_n;
    reg  [IN_WIDTH-1:0]     i_input_u;
    reg  [IN_WIDTH-1:0]     i_input_b;
    wire [`GEN_LANES_A-1:0] o_output_u_a;
    wire [`GEN_LANES_B-1:0] o_output_u_b;
    wire [`GEN_LANES_C-1:0] o_output_u_c;
    wire [`GEN_LANES_D-1:0] o_output_u_d;
    wire [`GEN_LANES_A-1:0] o_output_b_a;
    wire [`GEN_LANES_B-1:0] o_output_b_b;
    wire [`GEN_LANES_C-1:0] o_output_b_c;
    wire [`GEN_LANES_D-1:0] o_output_b_d;

    // Held fixed-point operands: OUT_CHANNELS*K weight codes, then OUT_CHANNELS
    // bias codes. Generated from the model, so the DUT sees its exact operands.
    // A code is the probability both polarities compare against, so one table
    // drives every DUT.
    reg [OPW-1:0] operand [0:OP_COUNT-1];
    initial $readmemb("vec/conv_ugemm_operand.hex", operand);

    // Geometry d's railed codes: weights at the top code and bias at 0, so a
    // railed input gives its lanes a partial sum of K and a silent one gives 0.
    reg [OPW-1:0] operand_d [0:OP_COUNT-1];
    initial $readmemb("vec/conv_ugemm_operand_d.hex", operand_d);

    wire [W_WIDTH-1:0] i_weight;
    wire [B_WIDTH-1:0] i_bias;
    wire [W_WIDTH-1:0] i_weight_d;
    wire [B_WIDTH-1:0] i_bias_d;

    genvar channel, tap;
    generate
        for (channel = 0; channel < `GEN_OUT_CHANNELS; channel = channel + 1) begin : g_operand
            for (tap = 0; tap < K; tap = tap + 1) begin : g_weight
                assign i_weight[(channel*K + tap)*OPW +: OPW] = operand[channel*K + tap];
                assign i_weight_d[(channel*K + tap)*OPW +: OPW] = operand_d[channel*K + tap];
            end
            assign i_bias[channel*OPW +: OPW] =
                operand[`GEN_OUT_CHANNELS * K + channel];
            assign i_bias_d[channel*OPW +: OPW] =
                operand_d[`GEN_OUT_CHANNELS * K + channel];
        end
    endgenerate

    conv_ugemm_unipolar #(
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
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_u_a)
    );

    // HAS_BIAS = 0 drops the bias addend, so entry and the default scale shrink;
    // padding 0 leaves no tap on the pad stream.
    conv_ugemm_unipolar #(
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
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_u_b)
    );

    conv_ugemm_unipolar #(
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
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_u_c)
    );


    // An explicit scale below the fan-in, on railed operands. This is the only
    // geometry whose accumulator moves, so it is the one the saturation block
    // reads WIDTH out of.

    conv_ugemm_unipolar #(
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
        .i_weight      (i_weight_d),
        .i_bias        (i_bias_d),
        .o_output      (o_output_u_d)
    );

    conv_ugemm_bipolar #(
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
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_b_a)
    );

    conv_ugemm_bipolar #(
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
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_b_b)
    );

    conv_ugemm_bipolar #(
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
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_b_c)
    );


    conv_ugemm_bipolar #(
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
        .i_weight      (i_weight_d),
        .i_bias        (i_bias_d),
        .o_output      (o_output_b_d)
    );

    // One character wider than the widest golden column, which the generator
    // asserts is geometry a's lane count: $fscanf("%s") truncates to the token
    // width, so a column read into a reg exactly as wide as it should be
    // saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    // The widest of every column is taken here instead of assumed.
    localparam integer WIDEST_ABC  = (`GEN_LANES_A > `GEN_LANES_B)
                                    ? ((`GEN_LANES_A > `GEN_LANES_C) ? `GEN_LANES_A : `GEN_LANES_C)
                                    : ((`GEN_LANES_B > `GEN_LANES_C) ? `GEN_LANES_B : `GEN_LANES_C);
    localparam integer WIDEST_LANE = (WIDEST_ABC > `GEN_LANES_D) ? WIDEST_ABC : `GEN_LANES_D;
    localparam integer MAX_CHARS   = ((IN_WIDTH > WIDEST_LANE) ? IN_WIDTH : WIDEST_LANE) + 1;

    integer fd, code, n, fails;
    reg                     rst;
    reg  [1023:0]           hdr_line;
    reg  [MAX_CHARS*8-1:0]  tok_in_u;
    reg  [MAX_CHARS*8-1:0]  tok_in_b;
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


    // Characters $fscanf("%s") stored: the bit width the golden row carries.
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

    initial begin
        i_clk           = 1'b0;
        i_rst_n         = 1'b1;
        i_input_u = {IN_WIDTH{1'b0}};
        i_input_b = {IN_WIDTH{1'b0}};

        fd = $fopen("vec/conv_ugemm.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/conv_ugemm.vec (run `make vectors` first)");
            $finish;
        end

        // skip the header line
        code = $fgets(hdr_line, fd);

        n = 0;
        fails = 0;
        // Each row is compared before the posedge, so this testbench can only drive a
        // pp_delay of 0; that pre-posedge sampling is what enforces it. This check
        // rejects a model whose pp_delay stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL conv_ugemm: testbench samples combinationally, model pp_delay is %0d",
                     `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // The loop ends on the first row that does not yield all 11 columns, so a
        // scan that stops consuming ends the run instead of spinning on $feof.
        code = $fscanf(fd, "%d %s %s %s %s %s %s %s %s %s %s\n", rst, tok_in_u, tok_in_b,
                       tok_u_a, tok_u_b, tok_u_c, tok_u_d, tok_b_a, tok_b_b, tok_b_c, tok_b_d);
        while (code == 11) begin
            begin : g_row
                check_width(tok_in_u, IN_WIDTH, "in_u");
                check_width(tok_in_b, IN_WIDTH, "in_b");
                check_width(tok_u_a, `GEN_LANES_A, "out_u_a");
                check_width(tok_u_b, `GEN_LANES_B, "out_u_b");
                check_width(tok_u_c, `GEN_LANES_C, "out_u_c");
                check_width(tok_u_d, `GEN_LANES_D, "out_u_d");
                check_width(tok_b_a, `GEN_LANES_A, "out_b_a");
                check_width(tok_b_b, `GEN_LANES_B, "out_b_b");
                check_width(tok_b_c, `GEN_LANES_C, "out_b_c");
                check_width(tok_b_d, `GEN_LANES_D, "out_b_d");

                // reset boundary: clear every sequence index, counter and accumulator
                if (rst == 1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                i_input_u = token_bits(tok_in_u);
                i_input_b = token_bits(tok_in_b);
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

                // clock edge advances the sequence indices, the counters and the accumulators
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end

            code = $fscanf(fd, "%d %s %s %s %s %s %s %s %s %s %s\n", rst, tok_in_u, tok_in_b,
                           tok_u_a, tok_u_b, tok_u_c, tok_u_d, tok_b_a, tok_b_b, tok_b_c, tok_b_d);
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL conv_ugemm: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS conv_ugemm: %0d/%0d vectors (%0d, %0d, %0d and %0d lanes x %0d taps, scale %0d, %0d, %0d and %0d)",
                     n, n, `GEN_LANES_A, `GEN_LANES_B, `GEN_LANES_C, `GEN_LANES_D, K,
                     `GEN_SCALE_A, `GEN_SCALE_B, `GEN_SCALE_C, `GEN_SCALE_D);
        else
            $display("FAIL conv_ugemm: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
