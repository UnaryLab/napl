`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear (bipolar) over LANES output features.
// Each lane is one output feature: the layer's per-timestep partial sum
// 2*sum(x*w) - sum(x) - sum(w) + IN_FEATURES is, term by term,
// sum_f (1 - x_f - w_f + 2*x_f*w_f) = sum_f XNOR(x_f, w_f), so the partial sum is
// the popcount of the bipolar Gaines products. add_any popcounts its ENTRY-bit
// input, so the lane is IN_FEATURES mul_gaines_bipolar cells plus the optional
// bias spike feeding one add_any_bipolar. Those are operation-layer circuits and
// are instantiated rather than rebuilt. Generated parameters mirror Python.
// Weight and bias spikes arrive on ports: the model re-encodes both every
// timestep from externally held tensors.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every accumulator to match reset().
// WIDTH must satisfy 2**(WIDTH-1) > ENTRY (= IN_FEATURES + HAS_BIAS), so the
// signed accumulator holds a partial sum; the generate guard below enforces it
// at elaboration and mapping.yaml carries the same restriction.


module linear_bipolar #(
    parameter integer IN_FEATURES = 16,  // weight columns;     tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;    tb overrides via `GEN_LANES
    parameter integer WIDTH       = 12,  // accumulator width;  tb overrides via `GEN_WIDTH
    parameter integer SCALE       = 17,  // output divisor;     tb overrides via `GEN_SCALE
    parameter integer HAS_BIAS    = 1    // 1 adds a bias spike addend, 0 drops it
) (
    input  wire                         i_clk,
    input  wire                         i_rst_n,
    input  wire [IN_FEATURES-1:0]       i_input_spike,
    input  wire [LANES*IN_FEATURES-1:0] i_weight,  // lane l, feature f at [l*IN_FEATURES + f]
    input  wire [LANES-1:0]             i_bias,    // lane l at [l]
    output wire [LANES-1:0]             o_out
);


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
            ERROR_linear_WIDTH_too_small_for_ENTRY u_bad ();
        end
    endgenerate

    wire [ENTRY-1:0] addend [0:LANES-1];

    genvar lane, feature;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            for (feature = 0; feature < IN_FEATURES; feature = feature + 1) begin : g_mul
                mul_gaines_bipolar u_mul (
                    .i_input_0 (i_input_spike[feature]),
                    .i_input_1 (i_weight[lane*IN_FEATURES + feature]),
                    .o_out     (addend[lane][feature])
                );
            end

            if (HAS_BIAS != 0) begin : g_bias
                assign addend[lane][ENTRY-1] = i_bias[lane];
            end

            add_any_bipolar #(
                .SCALE (SCALE),
                .WIDTH (WIDTH),
                .ENTRY (ENTRY)
            ) u_acc (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input (addend[lane]),
                .o_out   (o_out[lane])
            );
        end
    endgenerate
endmodule
`default_nettype wire
