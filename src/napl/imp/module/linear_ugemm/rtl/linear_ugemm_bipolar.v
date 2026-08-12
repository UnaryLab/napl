`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear_ugemm (bipolar) over LANES output features.
// Each lane is one output feature: IN_FEATURES conditionally generated weight
// bits, an optional free-running bias bit, and a scaled unary accumulator over
// the resulting ENTRY addends. Those are exactly the operation-layer circuits
// mul_ugemm_bipolar, encode, and add_scale_bipolar, so they are instantiated
// rather than rebuilt. Generated parameters mirror Python.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every sequence index and accumulator to match reset().
// WIDTH must satisfy 2**(WIDTH-1) > ENTRY (= IN_FEATURES + HAS_BIAS), so the
// signed accumulator holds a partial sum; the generate guard below enforces it
// at elaboration and mapping.yaml carries the same restriction.
//
// The Python model keeps one sequence index pair (the input-one and input-zero
// paths) per input feature, shared by the whole weight column. Each lane here
// holds its own pair per input feature; every copy is advanced by the same input
// spike, so all copies carry the same value and the outputs are bit-exact with
// the shared-index model.


module linear_ugemm_bipolar #(
    parameter integer IN_FEATURES = 16,  // weight columns;        tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;       tb overrides via `GEN_LANES
    parameter integer SEQ_WIDTH   = 8,   // ceil(log2(timestep));  tb overrides via `GEN_SEQ_WIDTH
    parameter integer WIDTH       = 12,  // accumulator width;     tb overrides via `GEN_WIDTH
    parameter integer SCALE       = 17,  // output divisor;        tb overrides via `GEN_SCALE
    parameter integer HAS_BIAS    = 1    // 1 encodes a bias addend, 0 drops it
) (
    input  wire                                       i_clk,
    input  wire                                       i_rst_n,
    input  wire [IN_FEATURES-1:0]                     i_input,
    input  wire [LANES*IN_FEATURES*(SEQ_WIDTH+1)-1:0] i_weight,  // lane l, feature f at [(l*IN_FEATURES+f)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [LANES*(SEQ_WIDTH+1)-1:0]             i_bias,    // lane l at [l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                           o_output
);


    localparam integer OPW   = SEQ_WIDTH + 1;             // fixed-point operand width
    localparam integer ENTRY = IN_FEATURES + HAS_BIAS;    // addends per lane


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when the signed accumulator cannot hold a partial sum,
    // which is the Python constructor's 2**(width-1) > entry check. The compare
    // is on widths, so it stays exact past the 32-bit range of 2**(WIDTH-1).
    generate
        if (WIDTH - 1 < clog2(ENTRY + 1)) begin : g_bad_width
            ERROR_linear_ugemm_WIDTH_too_small_for_ENTRY u_bad ();
        end
    endgenerate

    wire [ENTRY-1:0] addend [0:LANES-1];

    genvar lane, feature;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            for (feature = 0; feature < IN_FEATURES; feature = feature + 1) begin : g_mul
                mul_ugemm_bipolar #(
                    .WIDTH (SEQ_WIDTH)
                ) u_mul (
                    .i_clk     (i_clk),
                    .i_rst_n   (i_rst_n),
                    .i_input_0 (i_input[feature]),
                    .i_input_1 (i_weight[(lane*IN_FEATURES + feature)*OPW +: OPW]),
                    .o_output  (addend[lane][feature])
                );
            end

            // The bias bit advances once per timestep, so it is the plain encoder
            // over the same number sequence.
            if (HAS_BIAS != 0) begin : g_bias
                encode #(
                    .WIDTH (SEQ_WIDTH),
                    .FRAC  (SEQ_WIDTH)
                ) u_bias (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_bias[lane*OPW +: OPW]),
                    .o_spike (addend[lane][ENTRY-1])
                );
            end

            add_scale_bipolar #(
                .SCALE (SCALE),
                .WIDTH (WIDTH),
                .ENTRY (ENTRY)
            ) u_acc (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input (addend[lane]),
                .o_output   (o_output[lane])
            );
        end
    endgenerate
endmodule
`default_nettype wire
