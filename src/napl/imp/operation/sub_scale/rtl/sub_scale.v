`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sub_scale -- bipolar scaled difference (a - b) / scale (clocked).
//
// RTL counterpart of napl.sim.operation.sub_scale
// (src/napl/sim/operation/sub_scale.py). The class computes the signed scaled
// difference as negate-then-scaled-add: it negates the subtrahend stream b and
// reduces {a, -b} with a scaled two-input adder, so (a - b) / scale equals
// (a + (-b)) / scale. Bipolar only, so one circuit covers every legal polarity
// and the module carries no polarity postfix.
//
// Composition (RULE_IMP module-layer reuse): instantiate the standalone
// operation circuits rather than reinlining their math. Both resolve from the
// operation/*/rtl -y library the co-sim compiles with.
//   - negate:            o = ~i, flips b's sign  (combinational, pp_delay 0)
//   - add_scale_bipolar: two-lane scaled adder    (combinational output, pp_delay 0)
// ENTRY is structurally 2 (the two streams a and -b). SCALE and WIDTH inherit
// from the sim config (scale, intwidth); the sim's fracwidth has no hardware
// form, so mapping.yaml pins it to 0 (the add_scale integer grid).
//
// Timing: output is combinational (pp_delay 0 = negate 0 + add_scale 0); each
// posedge advances the inner accumulator one timestep. Active-low reset maps to
// reset() (the add_scale accumulator clears to 0).
//
// Verify from src/napl/imp/: conda run -n napl make test OP=sub_scale
//==============================================================================


module sub_scale #(
    parameter integer SCALE = 2,     // inherited from config['scale'];   tb overrides via `GEN_SCALE
    parameter integer WIDTH = 20     // inherited from config['intwidth']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,   // minuend a    (bipolar spike stream)
    input  wire i_input_1,   // subtrahend b (bipolar spike stream)
    output wire o_output     // scaled difference (a - b) / scale
);
    // The two reduction lanes match the sim's stack order (input_0, negate(b)):
    // lane 0 = a, lane 1 = -b. add_scale only popcounts the lanes, so the order
    // is cosmetic, but it mirrors the model.
    localparam integer ENTRY = 2;

    wire neg_b;
    negate u_negate (
        .i_input (i_input_1),
        .o_output(neg_b)
    );

    add_scale_bipolar #(
        .SCALE(SCALE),
        .WIDTH(WIDTH),
        .ENTRY(ENTRY)
    ) u_add_scale (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input ({neg_b, i_input_0}),
        .o_output(o_output)
    );
endmodule
`default_nettype wire
