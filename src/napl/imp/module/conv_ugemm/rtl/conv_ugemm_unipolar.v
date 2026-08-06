`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.conv_ugemm (unipolar) over LANES output pixel-channels.
// Each lane is one (batch, out_channel, out_row, out_col) position. The model's
// per-timestep partial sum is the plain count of the conditionally generated
// product spikes over the im2col patch, plus the optional bias spike, so the
// lane is K = IN_CHANNELS*KERNEL_H*KERNEL_W mul_ugemm_unipolar cells and one
// encode cell feeding one add_any_unipolar. Those are operation-layer circuits
// and are instantiated rather than rebuilt. Generated parameters mirror Python.
// The im2col patch is wiring: tap (ic, kh, kw) of lane (b, oc, oh, ow) reads
// input row oh*STRIDE + kh*DILATION - PADDING and column ow*STRIDE +
// kw*DILATION - PADDING.
// A tap outside the input reads pad_bit, the zero spike the model pads a
// unipolar stream with. The pad tap drives its mul_ugemm cell like any other
// tap: a zero spike gates the product off and holds the cell's sequence index.
// Weight and bias arrive as held fixed-point codes, not spikes: this layer
// generates both streams in hardware (Python internal_encode = True).
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every sequence index, the bias counters, and every
// accumulator, to match reset().
// WIDTH must satisfy 2**(WIDTH-1) > ENTRY (= K + HAS_BIAS), so the signed
// accumulator holds a partial sum, and LANES must equal the output positions the
// geometry produces. The generate guards below enforce both at elaboration and
// mapping.yaml carries the same restrictions.
//
// The Python model keeps one sequence-index pair (the input-one and input-zero
// paths) per (spatial output position, patch tap), shared by every output
// channel, because the patch spike broadcasts over the output channels. Each
// lane here holds its own pair per tap, so the OUT_CHANNELS lanes of one spatial
// position hold OUT_CHANNELS copies of the model's one pair. Both premises of
// that equality have exact sites. The index update reads only the patch spike
// and never the comparator: sim/operation/mul_ugemm.py:171 in the model,
// operation/mul_ugemm/rtl/mul_ugemm_unipolar.v in the RTL. And every copy of one
// tap is driven by the same x_bit, assigned once per (spatial position, tap) in
// the g_input / g_pad branch below and fanned out to the OUT_CHANNELS lanes. So
// all copies carry the same value and the outputs are bit-exact with the
// shared-index model.
// The per-lane bias encoder replicates on a shorter argument: its counter
// advances by one unconditionally every timestep (operation/encode/rtl/encode.v
// lines 37-42), so it is data-independent and every lane's copy holds the same
// index whatever that lane's inputs are.


module conv_ugemm_unipolar #(
    parameter integer BATCH        = 1,   // input batch;         tb overrides via `GEN_BATCH
    parameter integer IN_CHANNELS  = 1,   // input channels;      tb overrides via `GEN_IN_CHANNELS
    parameter integer IN_H         = 4,   // input rows;          tb overrides via `GEN_IN_H
    parameter integer IN_W         = 4,   // input columns;       tb overrides via `GEN_IN_W
    parameter integer OUT_CHANNELS = 2,   // output channels;     tb overrides via `GEN_OUT_CHANNELS
    parameter integer KERNEL_H     = 2,   // kernel rows;         tb overrides via `GEN_KERNEL_H
    parameter integer KERNEL_W     = 2,   // kernel columns;      tb overrides via `GEN_KERNEL_W
    parameter integer STRIDE       = 1,   // window step;         tb overrides via `GEN_STRIDE
    parameter integer PADDING      = 1,   // symmetric zero pad;  tb overrides via `GEN_PADDING
    parameter integer DILATION     = 1,   // kernel spacing;      tb overrides via `GEN_DILATION
    parameter integer SEQ_WIDTH    = 8,   // ceil(log2(timestep));tb overrides via `GEN_SEQ_WIDTH
    parameter integer WIDTH        = 12,  // accumulator width;   tb overrides via `GEN_WIDTH
    parameter integer SCALE        = 5,   // output divisor;      tb overrides via `GEN_SCALE
    parameter integer HAS_BIAS     = 1,   // 1 adds a bias addend, 0 drops it
    parameter integer LANES        = 50   // output positions;    tb overrides via `GEN_LANES
) (
    input  wire                                                                 i_clk,
    input  wire                                                                 i_rst_n,
    input  wire [BATCH*IN_CHANNELS*IN_H*IN_W-1:0]                               i_input_spike,  // (b, ic, ih, iw) row-major
    input  wire [OUT_CHANNELS*IN_CHANNELS*KERNEL_H*KERNEL_W*(SEQ_WIDTH+1)-1:0]  i_weight,       // out channel oc, tap t at [(oc*K + t)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [OUT_CHANNELS*(SEQ_WIDTH+1)-1:0]                                i_bias,         // out channel oc at [oc*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                                                     o_out           // (b, oc, oh, ow) row-major
);


    localparam integer OPW   = SEQ_WIDTH + 1;                      // fixed-point operand width
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
            ERROR_conv_ugemm_WIDTH_too_small_for_ENTRY u_bad ();
        end

        if (LANES != BATCH * OUT_CHANNELS * OUT_H * OUT_W) begin : g_bad_lanes
            ERROR_conv_ugemm_LANES_must_equal_OUTPUT_POSITIONS u_bad ();
        end
    endgenerate

    // Unipolar zero padding is a zero spike every timestep, so a padded tap
    // holds its cell's sequence index and contributes nothing.
    wire pad_bit;

    assign pad_bit = 1'b0;

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
                    assign x_bit = pad_bit;
                end

                mul_ugemm_unipolar #(
                    .WIDTH (SEQ_WIDTH)
                ) u_mul (
                    .i_clk     (i_clk),
                    .i_rst_n   (i_rst_n),
                    .i_input_0 (x_bit),
                    .i_input_1 (i_weight[(oc*K + TAP)*OPW +: OPW]),
                    .o_out     (addend[LANE][TAP])
                );
            end
            end
            end

            // The bias bit advances once per timestep on the same number
            // sequence, so it is the plain encoder over that sequence.
            if (HAS_BIAS != 0) begin : g_bias
                encode #(
                    .WIDTH (SEQ_WIDTH),
                    .FRAC  (SEQ_WIDTH)
                ) u_bias (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_bias[oc*OPW +: OPW]),
                    .o_spike (addend[LANE][ENTRY-1])
                );
            end

            add_any_unipolar #(
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
