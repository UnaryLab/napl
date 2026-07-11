`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// div_iscb_uni2bi -- unipolar-to-bipolar scaled add converter (stateful).
//
// RTL counterpart of napl.operation.uni2bi with width=3, as instantiated by
// div_iscb's bipolar_forward (src/napl/operation/uni2bi.py). Internal helper
// for div_iscb_bipolar; not a standalone op.
//
//   acc    = clamp(acc + (in + 1), -4, 3)     (updated this cycle, then read)
//   out    = (acc >= 2)
//   acc   -= 2*out
//
// Reset (active-low i_rst_n) loads the Python reset() state: acc = 0.
//==============================================================================
module div_iscb_uni2bi (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,
    output wire o_out
);
    // 5-bit signed accumulator holds [-4, 3] with slack for the +1/+2 step.
    localparam signed [4:0] ACC_MAX = 5'sd3;
    localparam signed [4:0] ACC_MIN = -5'sd4;

    reg signed [4:0] acc_q;

    // acc += (in + 1), clamped.
    wire signed [4:0] step    = i_in ? 5'sd2 : 5'sd1;
    wire signed [4:0] acc_sum = acc_q + step;
    wire signed [4:0] acc_clmp =
        (acc_sum > ACC_MAX) ? ACC_MAX :
        (acc_sum < ACC_MIN) ? ACC_MIN : acc_sum;

    wire out = (acc_clmp >= 5'sd2);    // acc >= 2
    assign o_out = out;

    // acc -= 2*out; stays within [-4, 3].
    wire signed [4:0] acc_next = out ? (acc_clmp - 5'sd2) : acc_clmp;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc_q <= 5'sd0;
        else
            acc_q <= acc_next;
    end
endmodule
`default_nettype wire
