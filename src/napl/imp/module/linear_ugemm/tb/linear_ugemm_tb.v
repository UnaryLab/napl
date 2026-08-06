`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "linear_ugemm/vec/linear_ugemm_params.vh"
// Python golden rows are <rst> <in_u> <in_b> <out_u> <out_u_nb> <out_b> <out_b_nb>,
// one per timestep. Polarity selects the circuit, so each row drives the unipolar
// and the bipolar DUT with its own encoded input stream, in the with-bias and the
// no-bias configuration. Outputs are combinational in the arrival cycle, so each
// row is checked before the posedge that advances the sequence indices and the
// accumulators. rst=1 pulses i_rst_n low first.
// Co-sim: make test MODULE=linear_ugemm


module linear_ugemm_tb;
    localparam integer OPW      = `GEN_SEQ_WIDTH + 1;
    localparam integer W_WIDTH  = `GEN_LANES * `GEN_IN_FEATURES * OPW;
    localparam integer B_WIDTH  = `GEN_LANES * OPW;
    localparam integer OP_COUNT = `GEN_LANES * (`GEN_IN_FEATURES + 1);

    reg                         i_clk;
    reg                         i_rst_n;
    reg  [`GEN_IN_FEATURES-1:0] i_input_spike_u;
    reg  [`GEN_IN_FEATURES-1:0] i_input_spike_b;
    wire [`GEN_LANES-1:0]       o_out_u;
    wire [`GEN_LANES-1:0]       o_out_u_nb;
    wire [`GEN_LANES-1:0]       o_out_b;
    wire [`GEN_LANES-1:0]       o_out_b_nb;

    // Held fixed-point operands: per lane, GEN_IN_FEATURES weight codes then the
    // bias code. Generated from the model, so the DUT sees its exact parameters.
    reg [OPW-1:0] operand_u [0:OP_COUNT-1];
    reg [OPW-1:0] operand_b [0:OP_COUNT-1];
    initial $readmemb("vec/linear_ugemm_operand_u.hex", operand_u);
    initial $readmemb("vec/linear_ugemm_operand_b.hex", operand_b);

    wire [W_WIDTH-1:0] i_weight_u;
    wire [W_WIDTH-1:0] i_weight_b;
    wire [B_WIDTH-1:0] i_bias_u;
    wire [B_WIDTH-1:0] i_bias_b;

    genvar lane, feature;
    generate
        for (lane = 0; lane < `GEN_LANES; lane = lane + 1) begin : g_operand
            for (feature = 0; feature < `GEN_IN_FEATURES; feature = feature + 1) begin : g_weight
                assign i_weight_u[(lane*`GEN_IN_FEATURES + feature)*OPW +: OPW] =
                    operand_u[lane*(`GEN_IN_FEATURES + 1) + feature];
                assign i_weight_b[(lane*`GEN_IN_FEATURES + feature)*OPW +: OPW] =
                    operand_b[lane*(`GEN_IN_FEATURES + 1) + feature];
            end
            assign i_bias_u[lane*OPW +: OPW] =
                operand_u[lane*(`GEN_IN_FEATURES + 1) + `GEN_IN_FEATURES];
            assign i_bias_b[lane*OPW +: OPW] =
                operand_b[lane*(`GEN_IN_FEATURES + 1) + `GEN_IN_FEATURES];
        end
    endgenerate

    linear_ugemm_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE),
        .HAS_BIAS    (1)
    ) dut_u (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u)
    );

    // HAS_BIAS = 0 drops the bias addend, so entry and the default scale shrink.
    linear_ugemm_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE_NB),
        .HAS_BIAS    (0)
    ) dut_u_nb (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u_nb)
    );

    linear_ugemm_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE),
        .HAS_BIAS    (1)
    ) dut_b (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_out         (o_out_b)
    );

    linear_ugemm_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .SEQ_WIDTH   (`GEN_SEQ_WIDTH),
        .WIDTH       (`GEN_WIDTH),
        .SCALE       (`GEN_SCALE_NB),
        .HAS_BIAS    (0)
    ) dut_b_nb (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_out         (o_out_b_nb)
    );

    integer fd, code, n, fails;
    reg                         rst;
    reg  [`GEN_IN_FEATURES-1:0] in_u;
    reg  [`GEN_IN_FEATURES-1:0] in_b;
    reg  [`GEN_LANES-1:0]       exp_u;
    reg  [`GEN_LANES-1:0]       exp_u_nb;
    reg  [`GEN_LANES-1:0]       exp_b;
    reg  [`GEN_LANES-1:0]       exp_b_nb;
    reg  [1023:0]               hdr_line;

    initial begin
        i_clk           = 1'b0;
        i_rst_n         = 1'b1;
        i_input_spike_u = {`GEN_IN_FEATURES{1'b0}};
        i_input_spike_b = {`GEN_IN_FEATURES{1'b0}};

        fd = $fopen("vec/linear_ugemm.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/linear_ugemm.vec (run `make vectors` first)");
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
            $display("FAIL linear_ugemm: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%d %b %b %b %b %b %b\n",
                           rst, in_u, in_b, exp_u, exp_u_nb, exp_b, exp_b_nb);
            if (code == 7) begin
                // reset boundary: clear every sequence index and accumulator
                if (rst == 1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                i_input_spike_u = in_u;
                i_input_spike_b = in_b;
                #1;

                n = n + 1;
                if (o_out_u !== exp_u) begin
                    $display("FAIL n=%0d unipolar bias : got %b exp %b", n, o_out_u, exp_u);
                    fails = fails + 1;
                end
                if (o_out_u_nb !== exp_u_nb) begin
                    $display("FAIL n=%0d unipolar nobias : got %b exp %b", n, o_out_u_nb, exp_u_nb);
                    fails = fails + 1;
                end
                if (o_out_b !== exp_b) begin
                    $display("FAIL n=%0d bipolar bias : got %b exp %b", n, o_out_b, exp_b);
                    fails = fails + 1;
                end
                if (o_out_b_nb !== exp_b_nb) begin
                    $display("FAIL n=%0d bipolar nobias : got %b exp %b", n, o_out_b_nb, exp_b_nb);
                    fails = fails + 1;
                end

                // clock edge advances the sequence indices and the accumulators
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS linear_ugemm: %0d/%0d vectors (%0d lanes x %0d in_features, scale %0d and %0d)",
                     n, n, `GEN_LANES, `GEN_IN_FEATURES, `GEN_SCALE, `GEN_SCALE_NB);
        else
            $display("FAIL linear_ugemm: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
