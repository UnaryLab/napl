`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// div_iscb_bi2uni -- bipolar-to-unipolar non-scaled add converter (stateful).
//
// RTL counterpart of napl.sim.operation.bi2uni with width=2, as instantiated by
// div_iscb's bipolar_forward (src/napl/sim/operation/bi2uni.py). Internal helper
// for div_iscb_bipolar; not a standalone op.
//
//   acc    = clamp(acc + (2*in - 1), -2, 1)   (updated this cycle, then read)
//   out    = (acc >= 1)
//   acc   -= out
//
// Reset (active-low i_rst_n) loads the Python reset() state: acc = 0.
//==============================================================================
module div_iscb_bi2uni (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,
    output wire o_out
);
    // 3-bit signed accumulator holds [-2, 1] with slack for the +/-1 step.
    localparam signed [3:0] ACC_MAX = 4'sd1;
    localparam signed [3:0] ACC_MIN = -4'sd2;

    reg signed [3:0] acc_q;

    // acc += 2*in - 1, clamped.
    wire signed [3:0] step    = i_in ? 4'sd1 : -4'sd1;
    wire signed [3:0] acc_sum = acc_q + step;
    wire signed [3:0] acc_clmp =
        (acc_sum > ACC_MAX) ? ACC_MAX :
        (acc_sum < ACC_MIN) ? ACC_MIN : acc_sum;

    wire out = (acc_clmp >= ACC_MAX);   // acc >= 1
    assign o_out = out;

    // acc -= out (carry-out removed); stays within [-2, 1].
    wire signed [3:0] acc_next = out ? (acc_clmp - 4'sd1) : acc_clmp;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc_q <= 4'sd0;
        else
            acc_q <= acc_next;
    end
endmodule
`default_nettype wire
