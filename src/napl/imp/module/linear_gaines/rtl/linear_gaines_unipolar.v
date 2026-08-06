`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear_gaines1 and linear_gaines2 (unipolar) over LANES output
// features. One module serves both classes: under a matched configuration the two
// build the same threshold sequences, so they differ only in the ROM image, in
// state the class never reads, and in their Python surface.
// Each lane is one output feature. The layer's per-timestep partial count is
// sum_f x_f * w_f plus the optional bias spike, so the lane is IN_FEATURES weight
// comparators feeding IN_FEATURES mul_gaines_unipolar cells, plus one encode cell
// for the bias.
// SCALED selects the Gaines adder at elaboration: the scaled arm compares the
// parallel count against the threshold sequence of an encode cell, which is the
// model's reference_encode(count / SCALE_LEN); the non-scaled arm emits
// count > 0, which is add_gaines held at SCALED = 0.
// Weight and bias arrive as held fixed-point codes, not spikes: this layer holds
// one threshold sequence per input feature and generates both streams itself
// (Python internal_encode = True).
// The weight-threshold ROM is W_LEN x IN_FEATURES and sits outside the lane
// generate, shared by every lane; the comparators cannot be shared with it,
// because each of the LANES x IN_FEATURES weight codes needs its own compare
// against the feature threshold of this timestep.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the threshold index and every encoder index, to match
// reset().
// SCALE_WIDTH must satisfy 2**(SCALE_WIDTH+1) > ENTRY (= IN_FEATURES + HAS_BIAS),
// so the threshold comparator input holds the parallel count; the generate guard
// below enforces it at elaboration.


module linear_gaines_unipolar #(
    parameter integer IN_FEATURES = 16,  // weight columns;        tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;       tb overrides via `GEN_LANES
    parameter integer SEQ_WIDTH   = 8,   // ceil(log2(timestep));  tb overrides via `GEN_SEQ_WIDTH
    parameter integer SCALE_WIDTH = 4,   // round(log2(ENTRY));    tb overrides via `GEN_SCALE_WIDTH
    parameter integer HAS_BIAS    = 1,   // 1 adds a bias addend, 0 drops it
    parameter integer SCALED      = 1    // 1 selects the scaled adder, 0 the OR adder
) (
    input  wire                                    i_clk,
    input  wire                                    i_rst_n,
    input  wire [IN_FEATURES-1:0]                  i_input_spike,
    input  wire [LANES*IN_FEATURES*(SEQ_WIDTH+1)-1:0] i_weight,  // lane l, feature f at [(l*IN_FEATURES + f)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [LANES*(SEQ_WIDTH+1)-1:0]          i_bias,       // lane l at [l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                        o_out
);


    localparam integer OPW     = SEQ_WIDTH + 1;          // fixed-point operand width
    localparam integer W_LEN   = 1 << SEQ_WIDTH;         // weight-threshold sequence length
    localparam integer ENTRY   = IN_FEATURES + HAS_BIAS; // addends per lane


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    localparam integer COUNT_W = clog2(ENTRY + 1);       // bits of the parallel count


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when the threshold comparator cannot hold the parallel
    // count. The compare is on widths, so it stays exact past the 32-bit range
    // of 2**(SCALE_WIDTH+1).
    generate
        if (SCALED != 0 && SCALE_WIDTH + 1 < COUNT_W) begin : g_bad_scale_width
            ERROR_linear_gaines_SCALE_WIDTH_too_small_for_ENTRY u_bad ();
        end
    endgenerate

    // Weight thresholds, one row per timestep and one field per input feature,
    // generated from the model's own w_num_seq. The row is shared by every lane.
    reg [IN_FEATURES*SEQ_WIDTH-1:0] w_rom [0:W_LEN-1];
    initial $readmemb("vec/linear_gaines_w.hex", w_rom);

    reg  [SEQ_WIDTH-1:0]            seq_idx;
    wire [IN_FEATURES*SEQ_WIDTH-1:0] w_row;

    assign w_row = w_rom[seq_idx];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            seq_idx <= {SEQ_WIDTH{1'b0}};
        else
            seq_idx <= seq_idx + {{(SEQ_WIDTH-1){1'b0}}, 1'b1};
    end

    genvar lane, feature, addend_bit;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            wire [ENTRY-1:0] addend;

            for (feature = 0; feature < IN_FEATURES; feature = feature + 1) begin : g_mul
                wire w_spike;

                // The model's inlined encoder comparison: the held weight code
                // against this timestep's threshold for this input feature.
                assign w_spike = i_weight[(lane*IN_FEATURES + feature)*OPW +: OPW]
                               > {1'b0, w_row[feature*SEQ_WIDTH +: SEQ_WIDTH]};

                mul_gaines_unipolar u_mul (
                    .i_input_0 (i_input_spike[feature]),
                    .i_input_1 (w_spike),
                    .o_out     (addend[feature])
                );
            end

            // The bias bit advances once per timestep on its own sequence, so it
            // is the plain encoder over that sequence.
            if (HAS_BIAS != 0) begin : g_bias
                encode #(
                    .WIDTH (SEQ_WIDTH),
                    .FRAC  (SEQ_WIDTH)
                ) u_bias (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_bias[lane*OPW +: OPW]),
                    .o_spike (addend[ENTRY-1])
                );
            end

            if (SCALED != 0) begin : g_scaled
                wire [COUNT_W-1:0] partial [0:ENTRY];
                wire [SCALE_WIDTH:0] count;

                assign partial[0] = {COUNT_W{1'b0}};

                for (addend_bit = 0; addend_bit < ENTRY; addend_bit = addend_bit + 1) begin : g_count
                    assign partial[addend_bit+1] = partial[addend_bit]
                        + {{(COUNT_W-1){1'b0}}, addend[addend_bit]};
                end

                // The count is the model's pc, and pc / SCALE_LEN on the encoder's
                // 1/SCALE_LEN grid is that same integer, so the encoder compares
                // the count itself against its threshold sequence.
                assign count = partial[ENTRY];

                encode #(
                    .WIDTH    (SCALE_WIDTH),
                    .FRAC     (SCALE_WIDTH),
                    .ROM_FILE ("vec/gaines_rom.hex")
                ) u_ref (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (count),
                    .o_spike (o_out[lane])
                );
            end else begin : g_or
                // Non-scaled unipolar addition is count > 0, the OR reduction
                // add_gaines implements at SCALED = 0.
                add_gaines #(
                    .SCALED (0),
                    .ENTRY  (ENTRY)
                ) u_add (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (addend),
                    .o_out   (o_out[lane])
                );
            end
        end
    endgenerate
endmodule
`default_nettype wire
