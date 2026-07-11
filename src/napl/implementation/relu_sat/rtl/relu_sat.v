`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// relu_sat -- saturating bipolar rate-coded ReLU (stateful).
//
// RTL counterpart of napl.operation.relu_sat (src/napl/operation/relu.py).
// relu_sat is bipolar/rate-coded only (no polarity variants), so the module is
// the bare op name. It chains two add_any accumulators (scale=1, width=3):
//   sub_1: shifts input [-1,1] -> [-1,0]   (offset = (entry-scale)/2 = 0.5)
//   add_1: shifts [-1,0] -> [0,1]          (offset = 0.5)
//
// Each add_any per timestep:  acc += (partial - offset); clamp; out = acc>=scale;
// acc -= scale*out.  The accumulators here are scaled by 2 so the 0.5 offsets are
// integers (half-units): A = 2*acc, integer, clamped to [-8, 6] (= 2*[-4, 3]).
//   sub stage: A_sub += 2*i_in - 1;        clamp; out_sub = A_sub >= 2; A_sub -= 2*out_sub
//   add stage: A_add += 2*out_sub + 1;     clamp; o_out   = A_add >= 2; A_add -= 2*out_add
//
// Output is combinational from the current accumulator state (same cycle as the
// input), so pp_delay = 0. One Python forward() timestep == one posedge i_clk;
// active-low i_rst_n reproduces reset() (both accumulators = 0).
//==============================================================================
module relu_sat (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,        // bipolar rate-coded input spike
    output wire o_out        // bipolar rate-coded ReLU output spike
);
    // Half-unit accumulators (2*acc): signed, range [-8, 6]; 5-bit signed holds it.
    reg signed [4:0] acc_sub;
    reg signed [4:0] acc_add;

    // ---- combinational: this cycle's outputs and next-cycle accumulator state ----
    // sub stage
    wire signed [5:0] sum_sub = $signed({acc_sub[4], acc_sub}) + (i_in ? 6'sd1 : -6'sd1);
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
