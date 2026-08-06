`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "conv_pc/vec/conv_pc_params.vh"
// Python golden rows are <in_u> <in_b> <w_u> <w_b> <bias_u> <bias_b> <pad_u>
// <pad_b> <out_u_a> <out_u_b> <out_u_c> <out_b_a> <out_b_b> <out_b_c>, one per
// timestep. Polarity selects the circuit, so each row drives the unipolar and the
// bipolar DUT with its own encoded streams, in each of the three geometries:
// a (padding 1, bias), b (padding 0, no bias), c (padding 2, stride 2,
// dilation 2, bias). conv_pc holds no accumulator, so the DUTs are combinational
// and carry no clock or reset: each row is applied and compared on the spot.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended
// silently by %b. An out_* column is a count bus, LANES counts of COUNT_W bits
// each, so its expected character count is LANES*COUNT_W rather than LANES.
// Co-sim: make test MODULE=conv_pc


module conv_pc_tb;
    localparam integer IN_WIDTH = `GEN_BATCH * `GEN_IN_CHANNELS * `GEN_IN_H * `GEN_IN_W;
    localparam integer W_WIDTH  = `GEN_OUT_CHANNELS * `GEN_IN_CHANNELS
                                  * `GEN_KERNEL_H * `GEN_KERNEL_W;
    // A lane emits a count, not a spike, so an output port is LANES*COUNT_W wide.
    localparam integer OUT_WIDTH_A = `GEN_LANES_A * `GEN_COUNT_W_A;
    localparam integer OUT_WIDTH_B = `GEN_LANES_B * `GEN_COUNT_W_B;
    localparam integer OUT_WIDTH_C = `GEN_LANES_C * `GEN_COUNT_W_C;

    reg  [IN_WIDTH-1:0]    i_input_spike_u;
    reg  [IN_WIDTH-1:0]    i_input_spike_b;
    reg  [W_WIDTH-1:0]     i_weight_u;
    reg  [W_WIDTH-1:0]     i_weight_b;
    reg  [`GEN_OUT_CHANNELS-1:0] i_bias_u;
    reg  [`GEN_OUT_CHANNELS-1:0] i_bias_b;
    reg                    i_pad_bits_b;
    wire [OUT_WIDTH_A-1:0] o_out_u_a;
    wire [OUT_WIDTH_B-1:0] o_out_u_b;
    wire [OUT_WIDTH_C-1:0] o_out_u_c;
    wire [OUT_WIDTH_A-1:0] o_out_b_a;
    wire [OUT_WIDTH_B-1:0] o_out_b_b;
    wire [OUT_WIDTH_C-1:0] o_out_b_c;

    conv_pc_unipolar #(
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
        .HAS_BIAS     (`GEN_HAS_BIAS_A),
        .COUNT_W      (`GEN_COUNT_W_A),
        .LANES        (`GEN_LANES_A)
    ) dut_u_a (
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u_a)
    );

    // HAS_BIAS = 0 drops the bias addend, so entry shrinks with it; padding 0
    // leaves no tap on the pad port.
    conv_pc_unipolar #(
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
        .HAS_BIAS     (`GEN_HAS_BIAS_B),
        .COUNT_W      (`GEN_COUNT_W_B),
        .LANES        (`GEN_LANES_B)
    ) dut_u_b (
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u_b)
    );

    conv_pc_unipolar #(
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
        .HAS_BIAS     (`GEN_HAS_BIAS_C),
        .COUNT_W      (`GEN_COUNT_W_C),
        .LANES        (`GEN_LANES_C)
    ) dut_u_c (
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u_c)
    );

    conv_pc_bipolar #(
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
        .HAS_BIAS     (`GEN_HAS_BIAS_A),
        .COUNT_W      (`GEN_COUNT_W_A),
        .LANES        (`GEN_LANES_A)
    ) dut_b_a (
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .i_pad_bits    (i_pad_bits_b),
        .o_out         (o_out_b_a)
    );

    conv_pc_bipolar #(
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
        .HAS_BIAS     (`GEN_HAS_BIAS_B),
        .COUNT_W      (`GEN_COUNT_W_B),
        .LANES        (`GEN_LANES_B)
    ) dut_b_b (
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .i_pad_bits    (i_pad_bits_b),
        .o_out         (o_out_b_b)
    );

    conv_pc_bipolar #(
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
        .HAS_BIAS     (`GEN_HAS_BIAS_C),
        .COUNT_W      (`GEN_COUNT_W_C),
        .LANES        (`GEN_LANES_C)
    ) dut_b_c (
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .i_pad_bits    (i_pad_bits_b),
        .o_out         (o_out_b_c)
    );

    // One character wider than the widest golden column: $fscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    // The count buses make the widest column LANES*COUNT_W rather than a lane
    // count, so the widest of every column is taken here instead of assumed.
    localparam integer WIDEST_IN  = (IN_WIDTH > W_WIDTH) ? IN_WIDTH : W_WIDTH;
    localparam integer WIDEST_AB  = (OUT_WIDTH_A > OUT_WIDTH_B) ? OUT_WIDTH_A : OUT_WIDTH_B;
    localparam integer WIDEST_OUT = (WIDEST_AB > OUT_WIDTH_C) ? WIDEST_AB : OUT_WIDTH_C;
    localparam integer MAX_CHARS  = ((WIDEST_IN > WIDEST_OUT) ? WIDEST_IN : WIDEST_OUT) + 1;

    integer fd, code, n, fails;
    reg  [1023:0]           hdr_line;
    reg  [MAX_CHARS*8-1:0]  tok_in_u;
    reg  [MAX_CHARS*8-1:0]  tok_in_b;
    reg  [MAX_CHARS*8-1:0]  tok_w_u;
    reg  [MAX_CHARS*8-1:0]  tok_w_b;
    reg  [MAX_CHARS*8-1:0]  tok_bias_u;
    reg  [MAX_CHARS*8-1:0]  tok_bias_b;
    reg  [MAX_CHARS*8-1:0]  tok_pad_b;
    reg  [MAX_CHARS*8-1:0]  tok_u_a;
    reg  [MAX_CHARS*8-1:0]  tok_u_b;
    reg  [MAX_CHARS*8-1:0]  tok_u_c;
    reg  [MAX_CHARS*8-1:0]  tok_b_a;
    reg  [MAX_CHARS*8-1:0]  tok_b_b;
    reg  [MAX_CHARS*8-1:0]  tok_b_c;
    reg  [OUT_WIDTH_A-1:0]  exp_u_a;
    reg  [OUT_WIDTH_B-1:0]  exp_u_b;
    reg  [OUT_WIDTH_C-1:0]  exp_u_c;
    reg  [OUT_WIDTH_A-1:0]  exp_b_a;
    reg  [OUT_WIDTH_B-1:0]  exp_b_b;
    reg  [OUT_WIDTH_C-1:0]  exp_b_c;


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
        i_input_spike_u = {IN_WIDTH{1'b0}};
        i_input_spike_b = {IN_WIDTH{1'b0}};
        i_weight_u      = {W_WIDTH{1'b0}};
        i_weight_b      = {W_WIDTH{1'b0}};
        i_bias_u        = {`GEN_OUT_CHANNELS{1'b0}};
        i_bias_b        = {`GEN_OUT_CHANNELS{1'b0}};
        i_pad_bits_b    = 1'b0;

        fd = $fopen("vec/conv_pc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/conv_pc.vec (run `make vectors` first)");
            $finish;
        end

        // skip the header line
        code = $fgets(hdr_line, fd);

        n = 0;
        fails = 0;
        // Each row is compared in the cycle it is applied, so this testbench can
        // only drive a pp_delay of 0. This check rejects a model whose pp_delay
        // stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL conv_pc: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // The loop ends on the first row that does not yield all 13 columns, so a
        // scan that stops consuming ends the run instead of spinning on $feof.
        code = $fscanf(fd, "%s %s %s %s %s %s %s %s %s %s %s %s %s\n",
                       tok_in_u, tok_in_b, tok_w_u, tok_w_b, tok_bias_u, tok_bias_b,
                       tok_pad_b,
                       tok_u_a, tok_u_b, tok_u_c, tok_b_a, tok_b_b, tok_b_c);
        while (code == 13) begin
            begin : g_row
                check_width(tok_in_u, IN_WIDTH, "in_u");
                check_width(tok_in_b, IN_WIDTH, "in_b");
                check_width(tok_w_u, W_WIDTH, "w_u");
                check_width(tok_w_b, W_WIDTH, "w_b");
                check_width(tok_bias_u, `GEN_OUT_CHANNELS, "bias_u");
                check_width(tok_bias_b, `GEN_OUT_CHANNELS, "bias_b");
                check_width(tok_pad_b, 1, "pad_b");
                check_width(tok_u_a, OUT_WIDTH_A, "out_u_a");
                check_width(tok_u_b, OUT_WIDTH_B, "out_u_b");
                check_width(tok_u_c, OUT_WIDTH_C, "out_u_c");
                check_width(tok_b_a, OUT_WIDTH_A, "out_b_a");
                check_width(tok_b_b, OUT_WIDTH_B, "out_b_b");
                check_width(tok_b_c, OUT_WIDTH_C, "out_b_c");

                i_input_spike_u = token_bits(tok_in_u);
                i_input_spike_b = token_bits(tok_in_b);
                i_weight_u      = token_bits(tok_w_u);
                i_weight_b      = token_bits(tok_w_b);
                i_bias_u        = token_bits(tok_bias_u);
                i_bias_b        = token_bits(tok_bias_b);
                i_pad_bits_b    = token_bits(tok_pad_b);
                exp_u_a         = token_bits(tok_u_a);
                exp_u_b         = token_bits(tok_u_b);
                exp_u_c         = token_bits(tok_u_c);
                exp_b_a         = token_bits(tok_b_a);
                exp_b_b         = token_bits(tok_b_b);
                exp_b_c         = token_bits(tok_b_c);
                #1;

                n = n + 1;
                if (o_out_u_a !== exp_u_a) begin
                    $display("FAIL n=%0d unipolar pad1 bias : got %b exp %b", n, o_out_u_a, exp_u_a);
                    fails = fails + 1;
                end
                if (o_out_u_b !== exp_u_b) begin
                    $display("FAIL n=%0d unipolar pad0 nobias : got %b exp %b", n, o_out_u_b, exp_u_b);
                    fails = fails + 1;
                end
                if (o_out_u_c !== exp_u_c) begin
                    $display("FAIL n=%0d unipolar strided : got %b exp %b", n, o_out_u_c, exp_u_c);
                    fails = fails + 1;
                end
                if (o_out_b_a !== exp_b_a) begin
                    $display("FAIL n=%0d bipolar pad1 bias : got %b exp %b", n, o_out_b_a, exp_b_a);
                    fails = fails + 1;
                end
                if (o_out_b_b !== exp_b_b) begin
                    $display("FAIL n=%0d bipolar pad0 nobias : got %b exp %b", n, o_out_b_b, exp_b_b);
                    fails = fails + 1;
                end
                if (o_out_b_c !== exp_b_c) begin
                    $display("FAIL n=%0d bipolar strided : got %b exp %b", n, o_out_b_c, exp_b_c);
                    fails = fails + 1;
                end
            end

            code = $fscanf(fd, "%s %s %s %s %s %s %s %s %s %s %s %s %s\n",
                           tok_in_u, tok_in_b, tok_w_u, tok_w_b, tok_bias_u, tok_bias_b,
                           tok_pad_b,
                           tok_u_a, tok_u_b, tok_u_c, tok_b_a, tok_b_b, tok_b_c);
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL conv_pc: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS conv_pc: %0d/%0d vectors (%0d, %0d and %0d lanes x %0d taps, %0d-bit counts)",
                     n, n, `GEN_LANES_A, `GEN_LANES_B, `GEN_LANES_C,
                     `GEN_IN_CHANNELS * `GEN_KERNEL_H * `GEN_KERNEL_W, `GEN_COUNT_W_A);
        else
            $display("FAIL conv_pc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
