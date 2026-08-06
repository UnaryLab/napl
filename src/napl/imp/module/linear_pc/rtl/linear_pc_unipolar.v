`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear_pc (unipolar) over LANES output features.
// Each lane is one output feature and emits the parallel count of that feature's
// addends, not a spike: the layer's per-timestep count sum_f x_f * w_f is the
// popcount of AND(x_f, w_f), the unipolar Gaines products, plus the optional
// bias spike.
// That is linear_unipolar with the add_any_unipolar accumulator removed and the
// popcount it reduces exposed, so the lane is IN_FEATURES mul_gaines_unipolar cells
// feeding an ENTRY-input popcount. mul_gaines_unipolar is an operation-layer
// circuit and is instantiated rather than rebuilt; the operation layer holds no
// standalone popcount, add_any bundling it with the scaled accumulator linear_pc
// does not have.
// Weight and bias spikes arrive on ports: the model re-encodes both every
// timestep from externally held tensors.
// The lane holds no state, so the module is combinational (pp_delay=0) and carries
// no clock or reset. o_out packs LANES counts of COUNT_W bits, lane l at
// [l*COUNT_W +: COUNT_W].
// COUNT_W must equal clog2(ENTRY + 1), ENTRY = IN_FEATURES + HAS_BIAS, so a lane
// count is neither truncated nor padded. The generate guard below enforces it at
// elaboration.


module linear_pc_unipolar #(
    parameter integer IN_FEATURES = 16,  // weight columns;      tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;     tb overrides via `GEN_LANES
    parameter integer HAS_BIAS    = 1,   // 1 adds a bias spike addend, 0 drops it
    parameter integer COUNT_W     = 5    // bits per lane count; tb overrides via `GEN_COUNT_W
) (
    input  wire [IN_FEATURES-1:0]       i_input_spike,
    input  wire [LANES*IN_FEATURES-1:0] i_weight,  // lane l, feature f at [l*IN_FEATURES + f]
    input  wire [LANES-1:0]             i_bias,    // lane l at [l]
    output wire [LANES*COUNT_W-1:0]     o_out      // lane l count at [l*COUNT_W +: COUNT_W]
);


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    localparam integer ENTRY = IN_FEATURES + HAS_BIAS;    // addends per lane


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build. The count bus width is a port width, so it is a parameter
    // rather than a body localparam and is checked here instead.
    generate
        if (COUNT_W != clog2(ENTRY + 1)) begin : g_bad_count_w
            ERROR_linear_pc_COUNT_W_must_equal_CLOG2_ENTRY u_bad ();
        end
    endgenerate

    wire [ENTRY-1:0] addend [0:LANES-1];

    genvar lane, feature, index;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            for (feature = 0; feature < IN_FEATURES; feature = feature + 1) begin : g_mul
                mul_gaines_unipolar u_mul (
                    .i_input_0 (i_input_spike[feature]),
                    .i_input_1 (i_weight[lane*IN_FEATURES + feature]),
                    .o_out     (addend[lane][feature])
                );
            end

            if (HAS_BIAS != 0) begin : g_bias
                assign addend[lane][ENTRY-1] = i_bias[lane];
            end

            // The lane's parallel count: the popcount linear feeds into add_any
            // and this module publishes instead.
            wire [COUNT_W-1:0] partial [0:ENTRY];
            assign partial[0] = {COUNT_W{1'b0}};

            for (index = 0; index < ENTRY; index = index + 1) begin : g_count
                assign partial[index+1] = partial[index]
                    + {{(COUNT_W-1){1'b0}}, addend[lane][index]};
            end

            assign o_out[lane*COUNT_W +: COUNT_W] = partial[ENTRY];
        end
    endgenerate
endmodule
`default_nettype wire
