`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.conv_gaines (unipolar) over LANES output pixel-channels.
// The Python class composes linear_gaines over the im2col patch, so this module
// wires the patch and instantiates one linear_gaines_unipolar per spatial output
// position, with LANES = OUT_CHANNELS lanes each. The Gaines multipliers, weight
// comparators, bias encoder and Gaines adder all live in that module and are not
// restated here.
// The im2col patch is wiring: tap (ic, kh, kw) of position (b, oh, ow) reads
// input row oh*STRIDE + kh*DILATION - PADDING and column ow*STRIDE +
// kw*DILATION - PADDING. A tap outside the input reads a zero spike, which is
// what unipolar zero padding is: the model pads the input tensor with 0.
// Weight and bias arrive as held fixed-point codes, not spikes: the composed
// layer holds one threshold sequence per patch tap and generates both streams
// itself (Python internal_encode = True). Their packing is the linear_gaines
// packing at IN_FEATURES = K, which is the (oc*K + tap) order used here.
// Each position holds its own copy of the threshold index, the bias encoder
// index and the Gaines select index. All three advance by one unconditionally
// every timestep, so every copy holds the value the model's single shared core
// holds, and the outputs are bit-exact with that shared-core model.
// SCALED selects the Gaines adder at elaboration, as in linear_gaines.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every index, to match reset().
// LANES must equal the output positions the geometry produces; the generate
// guard below enforces it at elaboration. SCALE_WIDTH must equal clog2(ENTRY),
// which the composed linear_gaines_unipolar guards, so mapping.yaml carries that
// restriction without a second guard here.


module conv_gaines_unipolar #(
    parameter integer BATCH        = 1,   // input batch;         tb overrides via `GEN_BATCH
    parameter integer IN_CHANNELS  = 1,   // input channels;      tb overrides via `GEN_IN_CHANNELS
    parameter integer IN_H         = 5,   // input rows;          tb overrides via `GEN_IN_H
    parameter integer IN_W         = 4,   // input columns;       tb overrides via `GEN_IN_W
    parameter integer OUT_CHANNELS = 2,   // output channels;     tb overrides via `GEN_OUT_CHANNELS
    parameter integer KERNEL_H     = 3,   // kernel rows;         tb overrides via `GEN_KERNEL_H
    parameter integer KERNEL_W     = 1,   // kernel columns;      tb overrides via `GEN_KERNEL_W
    parameter integer STRIDE       = 1,   // window step;         tb overrides via `GEN_STRIDE_A
    parameter integer PADDING      = 1,   // symmetric zero pad;  tb overrides via `GEN_PADDING_A
    parameter integer DILATION     = 1,   // kernel spacing;      tb overrides via `GEN_DILATION_A
    parameter integer SEQ_WIDTH    = 8,   // ceil(log2(timestep));tb overrides via `GEN_SEQ_WIDTH
    parameter integer SCALE_WIDTH  = 2,   // round(log2(ENTRY));  tb overrides via `GEN_SCALE_WIDTH
    parameter integer HAS_BIAS     = 1,   // 1 adds a bias addend, 0 drops it
    parameter integer SCALED       = 1,   // 1 selects the scaled adder, 0 the OR adder
    parameter integer LANES        = 60   // output positions;    tb overrides via `GEN_LANES_A
) (
    input  wire                                                                i_clk,
    input  wire                                                                i_rst_n,
    input  wire [BATCH*IN_CHANNELS*IN_H*IN_W-1:0]                              i_input,   // (b, ic, ih, iw) row-major
    input  wire [OUT_CHANNELS*IN_CHANNELS*KERNEL_H*KERNEL_W*(SEQ_WIDTH+1)-1:0] i_weight,  // out channel oc, tap t at [(oc*K + t)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [OUT_CHANNELS*(SEQ_WIDTH+1)-1:0]                               i_bias,    // out channel oc at [oc*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                                                    o_output   // (b, oc, oh, ow) row-major
);


    localparam integer K         = IN_CHANNELS * KERNEL_H * KERNEL_W;  // taps per output
    localparam integer OUT_H     = (IN_H + 2*PADDING - DILATION*(KERNEL_H-1) - 1) / STRIDE + 1;
    localparam integer OUT_W     = (IN_W + 2*PADDING - DILATION*(KERNEL_W-1) - 1) / STRIDE + 1;
    localparam integer POSITIONS = BATCH * OUT_H * OUT_W;              // im2col patches


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when the lane count is not the one the geometry produces.
    generate
        if (LANES != POSITIONS * OUT_CHANNELS) begin : g_bad_lanes
            ERROR_conv_gaines_LANES_must_equal_OUTPUT_POSITIONS u_bad ();
        end
    endgenerate

    wire [K-1:0]            patch    [0:POSITIONS-1];
    wire [OUT_CHANNELS-1:0] position [0:POSITIONS-1];

    genvar b, oc, oh, ow, ic, kh, kw;
    generate
        for (b = 0; b < BATCH; b = b + 1) begin : g_batch
        for (oh = 0; oh < OUT_H; oh = oh + 1) begin : g_out_row
        for (ow = 0; ow < OUT_W; ow = ow + 1) begin : g_out_col
            localparam integer POS = (b*OUT_H + oh)*OUT_W + ow;

            for (ic = 0; ic < IN_CHANNELS; ic = ic + 1) begin : g_in_channel
            for (kh = 0; kh < KERNEL_H; kh = kh + 1) begin : g_tap_row
            for (kw = 0; kw < KERNEL_W; kw = kw + 1) begin : g_tap_col
                localparam integer TAP = (ic*KERNEL_H + kh)*KERNEL_W + kw;
                localparam integer IH  = oh*STRIDE + kh*DILATION - PADDING;
                localparam integer IW  = ow*STRIDE + kw*DILATION - PADDING;

                if (IH >= 0 && IH < IN_H && IW >= 0 && IW < IN_W) begin : g_input
                    assign patch[POS][TAP] =
                        i_input[((b*IN_CHANNELS + ic)*IN_H + IH)*IN_W + IW];
                end else begin : g_pad
                    assign patch[POS][TAP] = 1'b0;
                end
            end
            end
            end

            linear_gaines_unipolar #(
                .IN_FEATURES (K),
                .LANES       (OUT_CHANNELS),
                .SEQ_WIDTH   (SEQ_WIDTH),
                .SCALE_WIDTH (SCALE_WIDTH),
                .HAS_BIAS    (HAS_BIAS),
                .SCALED      (SCALED)
            ) u_core (
                .i_clk    (i_clk),
                .i_rst_n  (i_rst_n),
                .i_input  (patch[POS]),
                .i_weight (i_weight),
                .i_bias   (i_bias),
                .o_output (position[POS])
            );

            for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin : g_out_channel
                assign o_output[((b*OUT_CHANNELS + oc)*OUT_H + oh)*OUT_W + ow] =
                    position[POS][oc];
            end
        end
        end
        end
    endgenerate
endmodule
`default_nettype wire
