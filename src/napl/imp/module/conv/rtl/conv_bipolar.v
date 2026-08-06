`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.conv (bipolar) over LANES output pixel-channels.
// Each lane is one (batch, out_channel, out_row, out_col) position: the layer's
// per-timestep partial sum 2*sum(x*w) - sum(x) - sum(w) + K over the im2col patch
// is, term by term, sum_k (1 - x_k - w_k + 2*x_k*w_k) = sum_k XNOR(x_k, w_k), so
// the partial sum is the popcount of the bipolar Gaines products of the patch.
// add_any popcounts its ENTRY-bit input, so the lane is K = IN_CHANNELS*KERNEL_H*
// KERNEL_W mul_gaines_bipolar cells plus the optional bias spike feeding one
// add_any_bipolar. Those are operation-layer circuits and are instantiated rather
// than rebuilt. Generated parameters mirror Python.
// The im2col patch is wiring: tap (ic, kh, kw) of lane (b, oc, oh, ow) reads input
// row oh*STRIDE + kh*DILATION - PADDING and column ow*STRIDE + kw*DILATION -
// PADDING. A tap outside the input reads i_pad_bits, the decorrelated rate-0.5
// bipolar-zero pad stream the Python model encodes on its own sobol dimension; a
// deterministic pad toggle would correlate with the weight stream.
// Weight, bias, and pad spikes arrive on ports: the model re-encodes all three
// every timestep from externally held tensors.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every accumulator to match reset().
// WIDTH must satisfy 2**(WIDTH-1) > ENTRY (= K + HAS_BIAS), so the signed
// accumulator holds a partial sum, and LANES must equal the output positions the
// geometry produces. The generate guards below enforce both at elaboration and
// mapping.yaml carries the same restrictions.


module conv_bipolar #(
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
    parameter integer WIDTH        = 12,  // accumulator width;   tb overrides via `GEN_WIDTH
    parameter integer SCALE        = 19,  // output divisor;      tb overrides via `GEN_SCALE
    parameter integer HAS_BIAS     = 1,   // 1 adds a bias spike addend, 0 drops it
    parameter integer LANES        = 108  // output positions;    tb overrides via `GEN_LANES
) (
    input  wire                                                   i_clk,
    input  wire                                                   i_rst_n,
    input  wire [BATCH*IN_CHANNELS*IN_H*IN_W-1:0]                 i_input_spike,  // (b, ic, ih, iw) row-major
    input  wire [OUT_CHANNELS*IN_CHANNELS*KERNEL_H*KERNEL_W-1:0]  i_weight,       // out channel oc, tap t at [oc*K + t]
    input  wire [OUT_CHANNELS-1:0]                                i_bias,         // out channel oc at [oc]
    input  wire                                                   i_pad_bits,     // one pad spike per timestep, all lanes
    output wire [LANES-1:0]                                       o_out           // (b, oc, oh, ow) row-major
);


    localparam integer K     = IN_CHANNELS * KERNEL_H * KERNEL_W;  // taps per output
    localparam integer ENTRY = K + HAS_BIAS;                       // addends per lane
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
            ERROR_conv_WIDTH_too_small_for_ENTRY u_bad ();
        end

        if (LANES != BATCH * OUT_CHANNELS * OUT_H * OUT_W) begin : g_bad_lanes
            ERROR_conv_LANES_must_equal_OUTPUT_POSITIONS u_bad ();
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
                    assign x_bit = i_input_spike[((b*IN_CHANNELS + ic)*IN_H + IH)*IN_W + IW];
                end else begin : g_pad
                    assign x_bit = i_pad_bits;
                end

                mul_gaines_bipolar u_mul (
                    .i_input_0 (x_bit),
                    .i_input_1 (i_weight[oc*K + TAP]),
                    .o_out     (addend[LANE][TAP])
                );
            end
            end
            end

            if (HAS_BIAS != 0) begin : g_bias
                assign addend[LANE][ENTRY-1] = i_bias[oc];
            end

            add_any_bipolar #(
                .SCALE (SCALE),
                .WIDTH (WIDTH),
                .ENTRY (ENTRY)
            ) u_acc (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input (addend[LANE]),
                .o_out   (o_out[LANE])
            );
        end
        end
        end
        end
    endgenerate
endmodule
`default_nettype wire
