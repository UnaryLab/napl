`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.conv_mix (unipolar) over LANES output pixel-channels.
// Each lane is one (batch, out_channel, out_row, out_col) position: the layer's
// per-timestep partial sum sum_k x_k * w_k over the im2col patch is the popcount
// of AND(x_k, w_k), the unipolar Gaines products of the patch.
// add_scale popcounts its ENTRY-bit input, so the lane is K = IN_CHANNELS*KERNEL_H*
// KERNEL_W (encode -> mul_gaines_unipolar) cells plus the optional encoded bias spike
// feeding one add_scale_unipolar. Those are operation-layer circuits and are
// instantiated rather than rebuilt. Generated parameters mirror Python.
// The im2col patch is wiring: tap (ic, kh, kw) of lane (b, oc, oh, ow) reads input
// row oh*STRIDE + kh*DILATION - PADDING and column ow*STRIDE + kw*DILATION -
// PADDING. A tap outside the input reads a constant zero spike, the unipolar zero
// pad, so this variant carries no pad encoder: the Python model builds a pad stream
// only for a bipolar stream with nonzero padding, and mapping.yaml maps `pad_bits`
// to no port here.
// The weight and bias encoders are held here: i_weight and i_bias carry held
// fixed-point probability codes, and one encode cell per weight tap and per output
// channel's bias re-encodes them every timestep. Weight and bias read separate
// number-sequence ROMs (W_ROM, B_ROM).
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every sequence index and accumulator to match reset().
// WIDTH must satisfy 2**(WIDTH-1) > ENTRY (= K + HAS_BIAS), so the signed
// accumulator holds a partial sum, and LANES must equal the output positions the
// geometry produces. The generate guards below enforce both at elaboration and
// mapping.yaml carries the same restrictions.


module conv_mix_unipolar #(
    parameter integer BATCH        = 1,   // input batch;         tb overrides via `GEN_BATCH
    parameter integer IN_CHANNELS  = 2,   // input channels;      tb overrides via `GEN_IN_CHANNELS
    parameter integer IN_H         = 6,   // input rows;          tb overrides via `GEN_IN_H
    parameter integer IN_W         = 6,   // input columns;       tb overrides via `GEN_IN_W
    parameter integer OUT_CHANNELS = 3,   // output channels;     tb overrides via `GEN_OUT_CHANNELS
    parameter integer KERNEL_H     = 3,   // kernel rows;         tb overrides via `GEN_KERNEL_H
    parameter integer KERNEL_W     = 3,   // kernel columns;      tb overrides via `GEN_KERNEL_W
    parameter integer STRIDE       = 1,   // window step;         tb overrides via `GEN_STRIDE
    parameter integer PADDING      = 1,   // symmetric zero pad;  tb overrides via `GEN_PADDING
    parameter integer DILATION     = 1,   // kernel spacing;      tb overrides via `GEN_DILATION
    parameter integer SEQ_WIDTH    = 8,   // ceil(log2(timestep));tb overrides via `GEN_SEQ_WIDTH
    parameter integer WIDTH        = 12,  // accumulator width;   tb overrides via `GEN_WIDTH
    parameter integer SCALE        = 19,  // output divisor;      tb overrides via `GEN_SCALE
    parameter integer HAS_BIAS     = 1,   // 1 encodes a bias spike addend, 0 drops it
    parameter integer LANES        = 108, // output positions;    tb overrides via `GEN_LANES
    parameter W_ROM   = "vec/cm_wrom.hex",// weight number-sequence ROM, sim-cwd relative
    parameter B_ROM   = "vec/cm_brom.hex" // bias number-sequence ROM, sim-cwd relative
) (
    input  wire                                                       i_clk,
    input  wire                                                       i_rst_n,
    input  wire [BATCH*IN_CHANNELS*IN_H*IN_W-1:0]                     i_input,   // (b, ic, ih, iw) row-major
    input  wire [OUT_CHANNELS*IN_CHANNELS*KERNEL_H*KERNEL_W*(SEQ_WIDTH+1)-1:0] i_weight, // oc, tap t at [(oc*K + t)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [OUT_CHANNELS*(SEQ_WIDTH+1)-1:0]                      i_bias,    // oc at [oc*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                                           o_output   // (b, oc, oh, ow) row-major
);


    localparam integer OPW   = SEQ_WIDTH + 1;                     // fixed-point operand width
    localparam integer K     = IN_CHANNELS * KERNEL_H * KERNEL_W; // taps per output
    localparam integer ENTRY = K + HAS_BIAS;                      // addends per lane
    localparam integer OUT_H = (IN_H + 2*PADDING - DILATION*(KERNEL_H-1) - 1) / STRIDE + 1;
    localparam integer OUT_W = (IN_W + 2*PADDING - DILATION*(KERNEL_W-1) - 1) / STRIDE + 1;


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    // Elaboration-time guards: an unresolvable module reference makes iverilog
    // fail the build. The width compare is the Python constructor's
    // 2**(width-1) > entry check, written on widths so it stays exact past the
    // 32-bit range of 2**(WIDTH-1).
    generate
        if (WIDTH - 1 < clog2(ENTRY + 1)) begin : g_bad_width
            ERROR_conv_mix_WIDTH_too_small_for_ENTRY u_bad ();
        end

        if (LANES != BATCH * OUT_CHANNELS * OUT_H * OUT_W) begin : g_bad_lanes
            ERROR_conv_mix_LANES_must_equal_OUTPUT_POSITIONS u_bad ();
        end
    endgenerate

    wire [ENTRY-1:0] addend [0:LANES-1];

    genvar b, oc, oh, ow, ic, kh, kw;
    generate
        for (b = 0; b < BATCH; b = b + 1) begin : g_batch
        for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin : g_out_channel
        for (oh = 0; oh < OUT_H; oh = oh + 1) begin : g_out_row
        for (ow = 0; ow < OUT_W; ow = ow + 1) begin : g_out_col
            localparam integer LANE = ((b*OUT_CHANNELS + oc)*OUT_H + oh)*OUT_W + ow;

            for (ic = 0; ic < IN_CHANNELS; ic = ic + 1) begin : g_in_channel
            for (kh = 0; kh < KERNEL_H; kh = kh + 1) begin : g_tap_row
            for (kw = 0; kw < KERNEL_W; kw = kw + 1) begin : g_tap_col
                localparam integer TAP = (ic*KERNEL_H + kh)*KERNEL_W + kw;
                localparam integer IH  = oh*STRIDE + kh*DILATION - PADDING;
                localparam integer IW  = ow*STRIDE + kw*DILATION - PADDING;

                wire x_bit;
                if (IH >= 0 && IH < IN_H && IW >= 0 && IW < IN_W) begin : g_input
                    assign x_bit = i_input[((b*IN_CHANNELS + ic)*IN_H + IH)*IN_W + IW];
                end else begin : g_pad
                    assign x_bit = 1'b0;
                end

                // The weight code is re-encoded to a spike every timestep, then the
                // unipolar Gaines product with the patch spike is an AND.
                wire w_spike;
                encode #(
                    .WIDTH    (SEQ_WIDTH),
                    .FRAC     (SEQ_WIDTH),
                    .ROM_FILE (W_ROM)
                ) u_w_enc (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_weight[(oc*K + TAP)*OPW +: OPW]),
                    .o_spike (w_spike)
                );

                mul_gaines_unipolar u_mul (
                    .i_input_0 (x_bit),
                    .i_input_1 (w_spike),
                    .o_output  (addend[LANE][TAP])
                );
            end
            end
            end

            if (HAS_BIAS != 0) begin : g_bias
                encode #(
                    .WIDTH    (SEQ_WIDTH),
                    .FRAC     (SEQ_WIDTH),
                    .ROM_FILE (B_ROM)
                ) u_bias (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_bias[oc*OPW +: OPW]),
                    .o_spike (addend[LANE][ENTRY-1])
                );
            end

            add_scale_unipolar #(
                .SCALE (SCALE),
                .WIDTH (WIDTH),
                .ENTRY (ENTRY)
            ) u_acc (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input (addend[LANE]),
                .o_output   (o_output[LANE])
            );
        end
        end
        end
        end
    endgenerate
endmodule
`default_nettype wire
