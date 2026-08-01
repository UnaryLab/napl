`timescale 1ns/1ps
`default_nettype none
// Fixed-width bi2uni helper matching Python: clamp acc+(2*input-1) to [-2,1],
// emit at one, then subtract the emitted bit. Active-low reset clears acc.
module div_iscb_bi2uni (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    // 3-bit signed accumulator holds [-2, 1] with slack for the +/-1 step.
    localparam signed [3:0] ACC_MAX = 4'sd1;
    localparam signed [3:0] ACC_MIN = -4'sd2;

    reg signed [3:0] acc_q;

    // acc += 2*in - 1, clamped.
    wire signed [3:0] step    = i_input ? 4'sd1 : -4'sd1;
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
