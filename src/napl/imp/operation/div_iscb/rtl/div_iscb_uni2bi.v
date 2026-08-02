`timescale 1ns/1ps
`default_nettype none
// Fixed-width uni2bi helper matching Python: clamp acc+(input+1) to [-4,3],
// emit at two, then subtract twice the emitted bit. Active-low reset clears acc.


module div_iscb_uni2bi (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    // 5-bit signed accumulator holds [-4, 3] with slack for the +1/+2 step.
    localparam signed [4:0] ACC_MAX = 5'sd3;
    localparam signed [4:0] ACC_MIN = -5'sd4;

    reg signed [4:0] acc_q;

    // acc += (in + 1), clamped.
    wire signed [4:0] step    = i_input ? 5'sd2 : 5'sd1;
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
