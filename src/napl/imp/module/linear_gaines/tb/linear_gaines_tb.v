`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "linear_gaines/vec/linear_gaines_params.vh"
// Python golden rows are <rst> <in_u> <in_b> <out_u_a> <out_u_b> <out_u_c>
// <out_b_a>, one per timestep. The unipolar arms are a (scaled, bias),
// b (non-scaled, no bias), and c (non-scaled, bias). Bipolar addition supports
// only the scaled-with-bias arm. Weight and bias are held fixed-point codes read from
// vec/linear_gaines_operand.hex, and the weight, bias and threshold streams are
// built inside the DUT, so none of them is a vector column.
// Outputs are combinational in the arrival cycle, so each row is checked before
// the posedge that advances the threshold and encoder indices. rst=1 pulses
// i_rst_n low first.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended silently
// by %b.
// Co-sim: make test MODULE=linear_gaines


module linear_gaines_tb;
    localparam integer OPW      = `GEN_SEQ_WIDTH + 1;
    localparam integer W_WIDTH  = `GEN_LANES * `GEN_IN_FEATURES * OPW;
    localparam integer B_WIDTH  = `GEN_LANES * OPW;
    localparam integer OP_COUNT = `GEN_LANES * `GEN_IN_FEATURES + `GEN_LANES;

    reg                         i_clk;
    reg                         i_rst_n;
    reg  [`GEN_IN_FEATURES-1:0] i_input_u;
    reg  [`GEN_IN_FEATURES-1:0] i_input_b;
    wire [`GEN_LANES-1:0]       o_output_u_a;
    wire [`GEN_LANES-1:0]       o_output_u_b;
    wire [`GEN_LANES-1:0]       o_output_u_c;
    wire [`GEN_LANES-1:0]       o_output_b_a;

    // Held fixed-point operands: LANES*IN_FEATURES weight codes, then LANES bias
    // codes. Generated from the model, so the DUT sees its exact operands. A code
    // is the probability both polarities compare against, so one table drives
    // every DUT.
    reg [OPW-1:0] operand [0:OP_COUNT-1];
    initial $readmemb("vec/linear_gaines_operand.hex", operand);

    wire [W_WIDTH-1:0] i_weight;
    wire [B_WIDTH-1:0] i_bias;

    genvar lane, feature;
    generate
        for (lane = 0; lane < `GEN_LANES; lane = lane + 1) begin : g_operand
            for (feature = 0; feature < `GEN_IN_FEATURES; feature = feature + 1) begin : g_weight
                assign i_weight[(lane*`GEN_IN_FEATURES + feature)*OPW +: OPW] =
                    operand[lane*`GEN_IN_FEATURES + feature];
            end
            assign i_bias[lane*OPW +: OPW] =
                operand[`GEN_LANES * `GEN_IN_FEATURES + lane];
        end
    endgenerate

    linear_gaines_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH (`GEN_SCALE_WIDTH),
        .HAS_BIAS    (`GEN_HAS_BIAS_A),
        .SCALED      (`GEN_SCALED_A)
    ) dut_u_a (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_u_a)
    );

    // HAS_BIAS = 0 drops the bias addend, so the entry the threshold comparator
    // sizes against shrinks with it.
    linear_gaines_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH (`GEN_SCALE_WIDTH),
        .HAS_BIAS    (`GEN_HAS_BIAS_B),
        .SCALED      (`GEN_SCALED_B)
    ) dut_u_b (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_u_b)
    );

    // SCALED = 0 selects the OR adder, so the scaled threshold path is not the
    // only Gaines adder the co-simulation covers.
    linear_gaines_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH (`GEN_SCALE_WIDTH),
        .HAS_BIAS    (`GEN_HAS_BIAS_C),
        .SCALED      (`GEN_SCALED_C)
    ) dut_u_c (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_u),
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_u_c)
    );

    linear_gaines_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .SCALE_WIDTH (`GEN_SCALE_WIDTH),
        .HAS_BIAS    (`GEN_HAS_BIAS_BIPOLAR)
    ) dut_b_a (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input (i_input_b),
        .i_weight      (i_weight),
        .i_bias        (i_bias),
        .o_output      (o_output_b_a)
    );

    // One character wider than the widest golden column: $fscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    localparam integer MAX_CHARS =
        ((`GEN_IN_FEATURES > `GEN_LANES) ? `GEN_IN_FEATURES : `GEN_LANES) + 1;

    integer fd, code, n, fails;
    reg                         rst;
    reg  [`GEN_IN_FEATURES-1:0] in_u;
    reg  [`GEN_IN_FEATURES-1:0] in_b;
    reg  [`GEN_LANES-1:0]       exp_u_a;
    reg  [`GEN_LANES-1:0]       exp_u_b;
    reg  [`GEN_LANES-1:0]       exp_u_c;
    reg  [`GEN_LANES-1:0]       exp_b_a;
    reg  [1023:0]               hdr_line;
    reg  [MAX_CHARS*8-1:0]      tok_in_u;
    reg  [MAX_CHARS*8-1:0]      tok_in_b;
    reg  [MAX_CHARS*8-1:0]      tok_u_a;
    reg  [MAX_CHARS*8-1:0]      tok_u_b;
    reg  [MAX_CHARS*8-1:0]      tok_u_c;
    reg  [MAX_CHARS*8-1:0]      tok_b_a;


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

        fd = $fopen("vec/linear_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/linear_gaines.vec (run `make vectors` first)");
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
            $display("FAIL linear_gaines: testbench samples combinationally, model pp_delay is %0d",
                     `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // The loop ends on the first row that does not yield all 7 columns, so a
        // scan that stops consuming ends the run instead of spinning on $feof.
        code = $fscanf(fd, "%d %s %s %s %s %s %s\n", rst, tok_in_u, tok_in_b,
                       tok_u_a, tok_u_b, tok_u_c, tok_b_a);
        while (code == 7) begin
            begin : g_row
                check_width(tok_in_u, `GEN_IN_FEATURES, "in_u");
                check_width(tok_in_b, `GEN_IN_FEATURES, "in_b");
                check_width(tok_u_a, `GEN_LANES, "out_u_a");
                check_width(tok_u_b, `GEN_LANES, "out_u_b");
                check_width(tok_u_c, `GEN_LANES, "out_u_c");
                check_width(tok_b_a, `GEN_LANES, "out_b_a");

                // reset boundary: clear every sequence index
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
                exp_b_a         = token_bits(tok_b_a);
                #1;

                n = n + 1;
                if (o_output_u_a !== exp_u_a) begin
                    $display("FAIL n=%0d unipolar scaled bias : got %b exp %b", n, o_output_u_a, exp_u_a);
                    fails = fails + 1;
                end
                if (o_output_u_b !== exp_u_b) begin
                    $display("FAIL n=%0d unipolar non-scaled nobias : got %b exp %b", n, o_output_u_b, exp_u_b);
                    fails = fails + 1;
                end
                if (o_output_u_c !== exp_u_c) begin
                    $display("FAIL n=%0d unipolar non-scaled : got %b exp %b", n, o_output_u_c, exp_u_c);
                    fails = fails + 1;
                end
                if (o_output_b_a !== exp_b_a) begin
                    $display("FAIL n=%0d bipolar scaled bias : got %b exp %b", n, o_output_b_a, exp_b_a);
                    fails = fails + 1;
                end

                // clock edge advances the sequence indices
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end

            code = $fscanf(fd, "%d %s %s %s %s %s %s\n", rst, tok_in_u, tok_in_b,
                           tok_u_a, tok_u_b, tok_u_c, tok_b_a);
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL linear_gaines: consumed %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS linear_gaines: %0d/%0d vectors (%0d lanes x %0d in_features, scale width %0d)",
                     n, n, `GEN_LANES, `GEN_IN_FEATURES, `GEN_SCALE_WIDTH);
        else
            $display("FAIL linear_gaines: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
