`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear_gaines1 and linear_gaines2 (bipolar) over LANES output
// features. One module serves both classes: under a matched configuration the two
// build the same threshold sequences, so they differ only in the ROM image, in
// state the class never reads, and in their Python surface.
// Each lane is one output feature. The layer's per-timestep partial count
// 2*sum(x*w) - sum(x) - sum(w) + IN_FEATURES is, term by term,
// sum_f (1 - x_f - w_f + 2*x_f*w_f) = sum_f XNOR(x_f, w_f), so the lane is
// IN_FEATURES weight comparators feeding IN_FEATURES mul_gaines_bipolar cells,
// plus one encode cell for the bias spike, which is one more addend.
// SCALED selects the Gaines adder at elaboration: the scaled arm compares the
// parallel count against the threshold sequence of an encode cell, which is the
// model's reference_encode(count / SCALE_LEN); the non-scaled arm integrates
// 2*count - ENTRY in the DEPTH-bit saturating counter below and emits
// counter > 2**(DEPTH-1).
// Weight and bias arrive as held fixed-point codes, not spikes: this layer holds
// one threshold sequence per input feature and generates both streams itself
// (Python internal_encode = True).
// The weight-threshold ROM is W_LEN x IN_FEATURES and sits outside the lane
// generate, shared by every lane; the comparators cannot be shared with it,
// because each of the LANES x IN_FEATURES weight codes needs its own compare
// against the feature threshold of this timestep.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the threshold index and every encoder index and loads
// every counter with 2**(DEPTH-1), the post-reset() state of the model.
// SCALE_WIDTH must satisfy 2**(SCALE_WIDTH+1) > ENTRY (= IN_FEATURES + HAS_BIAS),
// so the threshold comparator input holds the parallel count; the generate guard
// below enforces it at elaboration.
// The counter's bounds CNT_MAX and CNT_HALF are 32-bit signed elaboration
// constants sliced into the SUM_W-wide datapath, so DEPTH is capped at 30: at 31
// the 2**DEPTH - 1 constant is itself negative and the high clamp inverts. The
// model's own depth is a small counter width, so no configuration reaches that.


module linear_gaines_bipolar #(
    parameter integer IN_FEATURES = 16,  // weight columns;        tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;       tb overrides via `GEN_LANES
    parameter integer SEQ_WIDTH   = 8,   // ceil(log2(timestep));  tb overrides via `GEN_SEQ_WIDTH
    parameter integer SCALE_WIDTH = 4,   // round(log2(ENTRY));    tb overrides via `GEN_SCALE_WIDTH
    parameter integer DEPTH       = 8,   // counter width;         tb overrides via `GEN_DEPTH
    parameter integer HAS_BIAS    = 1,   // 1 adds a bias addend, 0 drops it
    parameter integer SCALED      = 1    // 1 selects the scaled adder, 0 the counter
) (
    input  wire                                       i_clk,
    input  wire                                       i_rst_n,
    input  wire [IN_FEATURES-1:0]                     i_input_spike,
    input  wire [LANES*IN_FEATURES*(SEQ_WIDTH+1)-1:0] i_weight,  // lane l, feature f at [(l*IN_FEATURES + f)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [LANES*(SEQ_WIDTH+1)-1:0]             i_bias,    // lane l at [l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                           o_out
);


    localparam integer OPW   = SEQ_WIDTH + 1;            // fixed-point operand width
    localparam integer W_LEN = 1 << SEQ_WIDTH;           // weight-threshold sequence length
    localparam integer ENTRY = IN_FEATURES + HAS_BIAS;   // addends per lane


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    localparam integer COUNT_W  = clog2(ENTRY + 1);      // bits of the parallel count
    localparam integer CNT_MAX  = (2 ** DEPTH) - 1;      // counter clamp, model cnt_max
    localparam integer CNT_HALF = 2 ** (DEPTH - 1);      // reset value and decision level
    // Holds the pre-clamp sum, which spans [-ENTRY, CNT_MAX + ENTRY], plus a sign.
    localparam integer SUM_W    = DEPTH + COUNT_W + 2;


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

    reg  [SEQ_WIDTH-1:0]             seq_idx;
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
            wire [ENTRY-1:0]   addend;
            wire [COUNT_W-1:0] partial [0:ENTRY];

            assign partial[0] = {COUNT_W{1'b0}};

            for (feature = 0; feature < IN_FEATURES; feature = feature + 1) begin : g_mul
                wire w_spike;

                // The model's inlined encoder comparison: the held weight code
                // against this timestep's threshold for this input feature.
                assign w_spike = i_weight[(lane*IN_FEATURES + feature)*OPW +: OPW]
                               > {1'b0, w_row[feature*SEQ_WIDTH +: SEQ_WIDTH]};

                mul_gaines_bipolar u_mul (
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

            for (addend_bit = 0; addend_bit < ENTRY; addend_bit = addend_bit + 1) begin : g_count
                assign partial[addend_bit+1] = partial[addend_bit]
                    + {{(COUNT_W-1){1'b0}}, addend[addend_bit]};
            end

            if (SCALED != 0) begin : g_scaled
                wire [SCALE_WIDTH:0] count;

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
            end else begin : g_counter
                // Non-scaled bipolar addition integrates 2*count - ENTRY in a
                // DEPTH-bit counter that saturates at both ends, and emits
                // counter > CNT_HALF. The clamped value is this cycle's output
                // and next cycle's state, so the path stays combinational.
                reg  signed [SUM_W-1:0] cnt;
                wire signed [SUM_W-1:0] s_max;
                wire signed [SUM_W-1:0] s_half;
                wire signed [SUM_W-1:0] s_entry;
                // A signed zero: an unsigned literal here would make the low
                // clamp an unsigned compare, which never fires.
                wire signed [SUM_W-1:0] s_zero;
                wire signed [SUM_W-1:0] count_ext;
                wire signed [SUM_W-1:0] sum;
                wire signed [SUM_W-1:0] clamped;

                assign s_max     = CNT_MAX[SUM_W-1:0];
                assign s_half    = CNT_HALF[SUM_W-1:0];
                assign s_entry   = ENTRY[SUM_W-1:0];
                assign s_zero    = {SUM_W{1'b0}};
                assign count_ext = $signed({{(SUM_W-COUNT_W){1'b0}}, partial[ENTRY]});
                assign sum       = cnt + (count_ext <<< 1) - s_entry;
                assign clamped   = (sum > s_max)  ? s_max :
                                   (sum < s_zero) ? s_zero : sum;

                assign o_out[lane] = (clamped > s_half);

                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        cnt <= s_half;
                    else
                        cnt <= clamped;
                end
            end
        end
    endgenerate
endmodule
`default_nettype wire
