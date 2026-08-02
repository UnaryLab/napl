`timescale 1ns/1ps
`default_nettype none
// Bipolar relu_sat equivalent using two width-3 add_any stages at 2x scale.
// Both accumulators clamp to [-8,6] and use half-unit offsets.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears both accumulators.


module relu_sat (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // bipolar rate-coded input spike
    output wire o_out        // bipolar rate-coded ReLU output spike
);
    // Half-unit accumulators (2*acc): signed, range [-8, 6]; 5-bit signed holds it.
    reg signed [4:0] acc_sub;
    reg signed [4:0] acc_add;

    // ---- combinational: this cycle's outputs and next-cycle accumulator state ----
    // sub stage
    wire signed [5:0] sum_sub = $signed({acc_sub[4], acc_sub}) + (i_input ? 6'sd1 : -6'sd1);
    wire signed [5:0] clmp_sub = (sum_sub > 6'sd6)  ? 6'sd6  :
                                 (sum_sub < -6'sd8) ? -6'sd8 : sum_sub;
    wire out_sub = (clmp_sub >= 6'sd2);
    // clmp_sub is in [-8, 6] and out_sub subtracts 2 only when clmp_sub >= 2,
    // so nxt_sub stays in [-8, 4]: fits 5-bit signed exactly.
    wire signed [4:0] nxt_sub = out_sub ? (clmp_sub[4:0] - 5'sd2) : clmp_sub[4:0];

    // add stage (input partial = sub_1_out + 1 -> half-units 2*out_sub + 1)
    wire signed [5:0] sum_add = $signed({acc_add[4], acc_add}) + (out_sub ? 6'sd3 : 6'sd1);
    wire signed [5:0] clmp_add = (sum_add > 6'sd6)  ? 6'sd6  :
                                 (sum_add < -6'sd8) ? -6'sd8 : sum_add;
    wire out_add = (clmp_add >= 6'sd2);
    wire signed [4:0] nxt_add = out_add ? (clmp_add[4:0] - 5'sd2) : clmp_add[4:0];

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
