`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "linear_mix/vec/linear_mix_params.vh"
// Python golden rows are <rst> <in_u> <in_b> <out_u> <out_u_nb> <out_u_s> <out_b>
// <out_b_nb> <out_b_s>, one per timestep. Polarity selects the circuit, so each row
// drives the unipolar and the bipolar DUT with its own encoded input stream, in the
// with-bias and the no-bias configuration. Weight and bias are held fixed-point
// codes loaded once from the operand hex files, since the RTL holds the encoders and
// re-encodes them every timestep. Outputs are combinational in the arrival cycle, so
// each row is checked before the posedge that advances the sequence indices and the
// accumulators. rst=1 pulses i_rst_n low first.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended
// silently by %b.
// Co-sim: make test MODULE=linear_mix


module linear_mix_tb;
    localparam integer OPW      = `GEN_SEQ_WIDTH + 1;
    localparam integer W_WIDTH  = `GEN_LANES * `GEN_IN_FEATURES * OPW;
    localparam integer B_WIDTH  = `GEN_LANES * OPW;
    localparam integer OP_COUNT = `GEN_LANES * (`GEN_IN_FEATURES + 1);

    reg                         i_clk;
    reg                         i_rst_n;
    reg  [`GEN_IN_FEATURES-1:0] i_input_u;
    reg  [`GEN_IN_FEATURES-1:0] i_input_b;
    wire [`GEN_LANES-1:0]       o_output_u;
    wire [`GEN_LANES-1:0]       o_output_u_nb;
    wire [`GEN_LANES-1:0]       o_output_b;
    wire [`GEN_LANES-1:0]       o_output_b_nb;
    wire [`GEN_LANES-1:0]       o_output_u_s;
    wire [`GEN_LANES-1:0]       o_output_b_s;

    // Held fixed-point operands: per lane, GEN_IN_FEATURES weight codes then the
    // bias code. Generated from the model, so the DUT sees its exact parameters.
    reg [OPW-1:0] operand_u [0:OP_COUNT-1];
    reg [OPW-1:0] operand_b [0:OP_COUNT-1];
    // The scaled arms' railed operands: weights at the top code and the bias at
    // 0, which is what lets their accumulators reach the clamps WIDTH sets. An
    // operand code is the probability code either polarity compares against, so
    // one file serves both.
    reg [OPW-1:0] operand_s [0:OP_COUNT-1];
    initial $readmemb("vec/linear_mix_operand_u.hex", operand_u);
    initial $readmemb("vec/linear_mix_operand_b.hex", operand_b);
    initial $readmemb("vec/linear_mix_operand_s.hex", operand_s);

    wire [W_WIDTH-1:0] i_weight_u;
    wire [W_WIDTH-1:0] i_weight_b;
    wire [W_WIDTH-1:0] i_weight_s;
    wire [B_WIDTH-1:0] i_bias_u;
    wire [B_WIDTH-1:0] i_bias_b;
    wire [B_WIDTH-1:0] i_bias_s;

    genvar lane, feature;
    generate
        for (lane = 0; lane < `GEN_LANES; lane = lane + 1) begin : g_operand
            for (feature = 0; feature < `GEN_IN_FEATURES; feature = feature + 1) begin : g_weight
                assign i_weight_u[(lane*`GEN_IN_FEATURES + feature)*OPW +: OPW] =
                    operand_u[lane*(`GEN_IN_FEATURES + 1) + feature];
                assign i_weight_b[(lane*`GEN_IN_FEATURES + feature)*OPW +: OPW] =
                    operand_b[lane*(`GEN_IN_FEATURES + 1) + feature];
                assign i_weight_s[(lane*`GEN_IN_FEATURES + feature)*OPW +: OPW] =
                    operand_s[lane*(`GEN_IN_FEATURES + 1) + feature];
            end
            assign i_bias_u[lane*OPW +: OPW] =
                operand_u[lane*(`GEN_IN_FEATURES + 1) + `GEN_IN_FEATURES];
            assign i_bias_b[lane*OPW +: OPW] =
                operand_b[lane*(`GEN_IN_FEATURES + 1) + `GEN_IN_FEATURES];
            assign i_bias_s[lane*OPW +: OPW] =
                operand_s[lane*(`GEN_IN_FEATURES + 1) + `GEN_IN_FEATURES];
        end
    endgenerate

    linear_mix_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE),
        .HAS_BIAS    (1)
    ) dut_u (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_output      (o_output_u)
    );

    // HAS_BIAS = 0 drops the bias addend, so entry and the default scale shrink.
    linear_mix_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE_NB),
        .HAS_BIAS    (0)
    ) dut_u_nb (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_output      (o_output_u_nb)
    );

    linear_mix_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE),
        .HAS_BIAS    (1)
    ) dut_b (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_output      (o_output_b)
    );

    linear_mix_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE_NB),
        .HAS_BIAS    (0)
    ) dut_b_nb (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_output      (o_output_b_nb)
    );

    // SCALE differs from ENTRY here, so the divisor is exercised independently of
    // the fan-in it defaults to, and the lane accumulator drifts instead of being
    // confined to [0, ENTRY - 1]. These are the arms the saturation blocks drive
    // onto the clamps WIDTH sets.
    linear_mix_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE_S),
        .HAS_BIAS    (1)
    ) dut_u_s (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight_s),
        .i_bias        (i_bias_s),
        .o_output      (o_output_u_s)
    );

    linear_mix_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE_S),
        .HAS_BIAS    (1)
    ) dut_b_s (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight_s),
        .i_bias        (i_bias_s),
        .o_output      (o_output_b_s)
    );

    // One character wider than the widest golden column: $fscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    // The widest of every column is taken here instead of assumed.
    localparam integer MAX_CHARS = ((`GEN_IN_FEATURES > `GEN_LANES)
                                    ? `GEN_IN_FEATURES : `GEN_LANES) + 1;

    integer fd, code, n, fails;
    reg                         rst;
    reg  [`GEN_IN_FEATURES-1:0] in_u;
    reg  [`GEN_IN_FEATURES-1:0] in_b;
    reg  [`GEN_LANES-1:0]       exp_u;
    reg  [`GEN_LANES-1:0]       exp_u_nb;
    reg  [`GEN_LANES-1:0]       exp_b;
    reg  [`GEN_LANES-1:0]       exp_b_nb;
    reg  [`GEN_LANES-1:0]       exp_u_s;
    reg  [`GEN_LANES-1:0]       exp_b_s;
    reg  [1023:0]               hdr_line;
    reg  [MAX_CHARS*8-1:0]      tok_in_u;
    reg  [MAX_CHARS*8-1:0]      tok_in_b;
    reg  [MAX_CHARS*8-1:0]      tok_u;
    reg  [MAX_CHARS*8-1:0]      tok_u_nb;
    reg  [MAX_CHARS*8-1:0]      tok_u_s;
    reg  [MAX_CHARS*8-1:0]      tok_b;
    reg  [MAX_CHARS*8-1:0]      tok_b_nb;
    reg  [MAX_CHARS*8-1:0]      tok_b_s;


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
        i_input_u = {`GEN_IN_FEATURES{1'b0}};
        i_input_b = {`GEN_IN_FEATURES{1'b0}};

        fd = $fopen("vec/linear_mix.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/linear_mix.vec (run `make vectors` first)");
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
            $display("FAIL linear_mix: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // The loop ends on the first row that does not yield all 9 columns, so a
        // scan that stops consuming ends the run instead of spinning on $feof.
        code = $fscanf(fd, "%d %s %s %s %s %s %s %s %s\n", rst, tok_in_u, tok_in_b,
                       tok_u, tok_u_nb, tok_u_s, tok_b, tok_b_nb, tok_b_s);
        while (code == 9) begin
            begin : g_row
                check_width(tok_in_u, `GEN_IN_FEATURES, "in_u_bits");
                check_width(tok_in_b, `GEN_IN_FEATURES, "in_b_bits");
                check_width(tok_u, `GEN_LANES, "out_u");
                check_width(tok_u_nb, `GEN_LANES, "out_u_nb");
                check_width(tok_b, `GEN_LANES, "out_b");
                check_width(tok_b_nb, `GEN_LANES, "out_b_nb");
                check_width(tok_u_s, `GEN_LANES, "out_u_s");
                check_width(tok_b_s, `GEN_LANES, "out_b_s");
                in_u     = token_bits(tok_in_u);
                in_b     = token_bits(tok_in_b);
                exp_u    = token_bits(tok_u);
                exp_u_nb = token_bits(tok_u_nb);
                exp_b    = token_bits(tok_b);
                exp_b_nb = token_bits(tok_b_nb);
                exp_u_s  = token_bits(tok_u_s);
                exp_b_s  = token_bits(tok_b_s);

                // reset boundary: clear every sequence index and accumulator
                if (rst == 1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                i_input_u = in_u;
                i_input_b = in_b;
                #1;

                n = n + 1;
                if (o_output_u !== exp_u) begin
                    $display("FAIL n=%0d unipolar bias : got %b exp %b", n, o_output_u, exp_u);
                    fails = fails + 1;
                end
                if (o_output_u_nb !== exp_u_nb) begin
                    $display("FAIL n=%0d unipolar nobias : got %b exp %b", n, o_output_u_nb, exp_u_nb);
                    fails = fails + 1;
                end
                if (o_output_b !== exp_b) begin
                    $display("FAIL n=%0d bipolar bias : got %b exp %b", n, o_output_b, exp_b);
                    fails = fails + 1;
                end
                if (o_output_b_nb !== exp_b_nb) begin
                    $display("FAIL n=%0d bipolar nobias : got %b exp %b", n, o_output_b_nb, exp_b_nb);
                    fails = fails + 1;
                end
                if (o_output_u_s !== exp_u_s) begin
                    $display("FAIL n=%0d unipolar scale=%0d : got %b exp %b",
                             n, `GEN_SCALE_S, o_output_u_s, exp_u_s);
                    fails = fails + 1;
                end
                if (o_output_b_s !== exp_b_s) begin
                    $display("FAIL n=%0d bipolar scale=%0d : got %b exp %b",
                             n, `GEN_SCALE_S, o_output_b_s, exp_b_s);
                    fails = fails + 1;
                end

                // clock edge advances the sequence indices and the accumulators
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end

            code = $fscanf(fd, "%d %s %s %s %s %s %s %s %s\n", rst, tok_in_u, tok_in_b,
                           tok_u, tok_u_nb, tok_u_s, tok_b, tok_b_nb, tok_b_s);
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL linear_mix: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS linear_mix: %0d/%0d vectors (%0d lanes x %0d in_features, scale %0d, %0d and %0d)",
                     n, n, `GEN_LANES, `GEN_IN_FEATURES, `GEN_SCALE, `GEN_SCALE_NB, `GEN_SCALE_S);
        else
            $display("FAIL linear_mix: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
