`timescale 1ns/1ps
`default_nettype none
// Bipolar relu_sat equivalent using two width-3 add_any stages at 2x scale.
// Both accumulators use half-unit offsets; only acc_sub carries a clamp, at -8.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears both accumulators.
//
// Invariant 1: acc_sub <= 1, so sum_sub <= 2 and the sub stage needs no upper
// clamp. acc_sub is 0 after reset. Given acc_sub = a <= 1: on i_input the sum is
// a+1 <= 2, which fires only when a+1 >= 2, that is a = 1, and then subtracts 2
// for a next state of 0; when a < 1 the sum a+1 <= 1 does not fire and the next
// state is a+1 <= 1. On a zero input the sum is a-1 < a and does not fire, so
// the next state is a-1 <= 0. Every case lands at or below 1. Only the lower
// side is unbounded, since a zero input decrements without limit, so the -8
// clamp on sum_sub is live and stays.
//
// Invariant 2: out_sub is never high on two consecutive cycles. out_sub requires
// acc_sub = 1 by invariant 1, and that transition leaves acc_sub = 0, which by
// the same case analysis cannot fire on the next cycle.
//
// Invariant 3: 0 <= acc_add <= 2, so 0 <= sum_add <= 4 and the add stage needs
// no clamp on either side. acc_add is 0 after reset and the addend, 2*out_sub+1,
// is always positive, so the lower bound is immediate: a fire needs sum_add >= 2
// and subtracts exactly 2. For the upper bound, note that acc_add = 2 is only
// reachable from acc_add = 1 with an addend of 3, so by invariant 2 the next
// addend is 1: from acc_add = 0 the sum is 1 or 3, giving a next state of 1;
// from acc_add = 1 the sum is 2 or 4, giving 0 or 2; from acc_add = 2 the addend
// is 1, so the sum is 3, giving 1. The largest sum reached is therefore 4 < 6.


module relu_sat (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // bipolar rate-coded input spike
    output wire o_out        // bipolar rate-coded ReLU output spike
);
    // Half-unit accumulators (2*acc): signed, acc_sub in [-8, 1] and acc_add in
    // [0, 2]; 5-bit signed holds both.
    reg signed [4:0] acc_sub;
    reg signed [4:0] acc_add;

    // ---- combinational: this cycle's outputs and next-cycle accumulator state ----
    // sub stage
    wire signed [5:0] sum_sub = $signed({acc_sub[4], acc_sub}) + (i_input ? 6'sd1 : -6'sd1);
    wire signed [5:0] clmp_sub = (sum_sub < -6'sd8) ? -6'sd8 : sum_sub;
    wire out_sub = (clmp_sub >= 6'sd2);
    // clmp_sub is in [-8, 2] and out_sub subtracts 2 only when clmp_sub >= 2,
    // so nxt_sub stays in [-8, 1]: fits 5-bit signed exactly.
    wire signed [4:0] nxt_sub = out_sub ? (clmp_sub[4:0] - 5'sd2) : clmp_sub[4:0];

    // add stage (input partial = sub_1_out + 1 -> half-units 2*out_sub + 1)
    wire signed [5:0] sum_add = $signed({acc_add[4], acc_add}) + (out_sub ? 6'sd3 : 6'sd1);
    wire out_add = (sum_add >= 6'sd2);
    wire signed [4:0] nxt_add = out_add ? (sum_add[4:0] - 5'sd2) : sum_add[4:0];

    assign o_out = out_add;

    // ---- sequential: advance accumulators; i_rst_n low maps to reset() (acc = 0) ----
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            acc_sub <= 5'sd0;
            acc_add <= 5'sd0;
        end else begin
            acc_sub <= nxt_sub;
            acc_add <= nxt_add;
        end
    end
endmodule
`default_nettype wire
