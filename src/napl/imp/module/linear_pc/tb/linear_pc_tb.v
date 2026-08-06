`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "linear_pc/vec/linear_pc_params.vh"
// Python golden rows are <in_u> <in_b> <w_u> <w_b> <b_u> <b_b> <out_u>
// <out_u_nb> <out_b> <out_b_nb>, one per timestep. Polarity selects the circuit,
// so each row drives the unipolar and the bipolar DUT with its own encoded
// streams, with and without the bias addend. linear_pc holds no accumulator, so
// the DUTs are combinational and carry no clock or reset: each row is applied and
// compared on the spot.
// Each vector column is scanned as text and its character count is checked
// against the port width before it is converted to bits, so a row that is not
// exactly as wide as its port fails here instead of being zero-extended
// silently by %b. An out_* column is a count bus, LANES counts of COUNT_W bits
// each, so its expected character count is LANES*COUNT_W rather than LANES.
// Co-sim: make test MODULE=linear_pc


module linear_pc_tb;
    localparam integer W_WIDTH = `GEN_LANES * `GEN_IN_FEATURES;
    // A lane emits a count, not a spike, so an output port is LANES*COUNT_W wide.
    localparam integer OUT_WIDTH    = `GEN_LANES * `GEN_COUNT_W;
    localparam integer OUT_WIDTH_NB = `GEN_LANES * `GEN_COUNT_W_NB;

    reg  [`GEN_IN_FEATURES-1:0] i_input_spike_u;
    reg  [`GEN_IN_FEATURES-1:0] i_input_spike_b;
    reg  [W_WIDTH-1:0]          i_weight_u;
    reg  [W_WIDTH-1:0]          i_weight_b;
    reg  [`GEN_LANES-1:0]       i_bias_u;
    reg  [`GEN_LANES-1:0]       i_bias_b;
    wire [OUT_WIDTH-1:0]        o_out_u;
    wire [OUT_WIDTH_NB-1:0]     o_out_u_nb;
    wire [OUT_WIDTH-1:0]        o_out_b;
    wire [OUT_WIDTH_NB-1:0]     o_out_b_nb;

    linear_pc_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .HAS_BIAS    (1),
        .COUNT_W     (`GEN_COUNT_W)
    ) dut_u (
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u)
    );

    // HAS_BIAS = 0 drops the bias addend, so entry and the count width shrink with it.
    linear_pc_unipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .HAS_BIAS    (0),
        .COUNT_W     (`GEN_COUNT_W_NB)
    ) dut_u_nb (
        .i_input_spike (i_input_spike_u),
        .i_weight      (i_weight_u),
        .i_bias        (i_bias_u),
        .o_out         (o_out_u_nb)
    );

    linear_pc_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .HAS_BIAS    (1),
        .COUNT_W     (`GEN_COUNT_W)
    ) dut_b (
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_out         (o_out_b)
    );

    linear_pc_bipolar #(
        .IN_FEATURES (`GEN_IN_FEATURES),
        .LANES       (`GEN_LANES),
        .HAS_BIAS    (0),
        .COUNT_W     (`GEN_COUNT_W_NB)
    ) dut_b_nb (
        .i_input_spike (i_input_spike_b),
        .i_weight      (i_weight_b),
        .i_bias        (i_bias_b),
        .o_out         (o_out_b_nb)
    );

    // One character wider than the widest golden column: $fscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    // The count buses put an out_* column at LANES*COUNT_W rather than a lane
    // count, so the widest of every column is taken here instead of assumed.
    localparam integer WIDEST_OUT = (OUT_WIDTH > OUT_WIDTH_NB) ? OUT_WIDTH : OUT_WIDTH_NB;
    localparam integer MAX_CHARS  = ((W_WIDTH > WIDEST_OUT) ? W_WIDTH : WIDEST_OUT) + 1;

    integer fd, code, n, fails;
    reg  [1023:0]           hdr_line;
    reg  [MAX_CHARS*8-1:0]  tok_in_u;
    reg  [MAX_CHARS*8-1:0]  tok_in_b;
    reg  [MAX_CHARS*8-1:0]  tok_w_u;
    reg  [MAX_CHARS*8-1:0]  tok_w_b;
    reg  [MAX_CHARS*8-1:0]  tok_b_u;
    reg  [MAX_CHARS*8-1:0]  tok_b_b;
    reg  [MAX_CHARS*8-1:0]  tok_out_u;
    reg  [MAX_CHARS*8-1:0]  tok_out_u_nb;
    reg  [MAX_CHARS*8-1:0]  tok_out_b;
    reg  [MAX_CHARS*8-1:0]  tok_out_b_nb;
    reg  [OUT_WIDTH-1:0]    exp_u;
    reg  [OUT_WIDTH_NB-1:0] exp_u_nb;
    reg  [OUT_WIDTH-1:0]    exp_b;
    reg  [OUT_WIDTH_NB-1:0] exp_b_nb;


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
        i_input_spike_u = {`GEN_IN_FEATURES{1'b0}};
        i_input_spike_b = {`GEN_IN_FEATURES{1'b0}};
        i_weight_u      = {W_WIDTH{1'b0}};
        i_weight_b      = {W_WIDTH{1'b0}};
        i_bias_u        = {`GEN_LANES{1'b0}};
        i_bias_b        = {`GEN_LANES{1'b0}};

        fd = $fopen("vec/linear_pc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/linear_pc.vec (run `make vectors` first)");
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
            $display("FAIL linear_pc: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // The loop ends on the first row that does not yield all 10 columns, so a
        // scan that stops consuming ends the run instead of spinning on $feof.
        code = $fscanf(fd, "%s %s %s %s %s %s %s %s %s %s\n",
                       tok_in_u, tok_in_b, tok_w_u, tok_w_b, tok_b_u, tok_b_b,
                       tok_out_u, tok_out_u_nb, tok_out_b, tok_out_b_nb);
        while (code == 10) begin
            begin : g_row
                check_width(tok_in_u, `GEN_IN_FEATURES, "in_u");
                check_width(tok_in_b, `GEN_IN_FEATURES, "in_b");
                check_width(tok_w_u, W_WIDTH, "w_u");
                check_width(tok_w_b, W_WIDTH, "w_b");
                check_width(tok_b_u, `GEN_LANES, "b_u");
                check_width(tok_b_b, `GEN_LANES, "b_b");
                check_width(tok_out_u, OUT_WIDTH, "out_u");
                check_width(tok_out_u_nb, OUT_WIDTH_NB, "out_u_nb");
                check_width(tok_out_b, OUT_WIDTH, "out_b");
                check_width(tok_out_b_nb, OUT_WIDTH_NB, "out_b_nb");

                i_input_spike_u = token_bits(tok_in_u);
                i_input_spike_b = token_bits(tok_in_b);
                i_weight_u      = token_bits(tok_w_u);
                i_weight_b      = token_bits(tok_w_b);
                i_bias_u        = token_bits(tok_b_u);
                i_bias_b        = token_bits(tok_b_b);
                exp_u           = token_bits(tok_out_u);
                exp_u_nb        = token_bits(tok_out_u_nb);
                exp_b           = token_bits(tok_out_b);
                exp_b_nb        = token_bits(tok_out_b_nb);
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
            end

            code = $fscanf(fd, "%s %s %s %s %s %s %s %s %s %s\n",
                           tok_in_u, tok_in_b, tok_w_u, tok_w_b, tok_b_u, tok_b_b,
                           tok_out_u, tok_out_u_nb, tok_out_b, tok_out_b_nb);
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's.
        if (n != `GEN_VECTORS) begin
            $display("FAIL linear_pc: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS linear_pc: %0d/%0d vectors (%0d lanes x %0d features, %0d-bit counts)",
                     n, n, `GEN_LANES, `GEN_IN_FEATURES, `GEN_COUNT_W);
        else
            $display("FAIL linear_pc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
