`timescale 1ns/1ps
`default_nettype none
// Fixed-width bi2uni helper matching Python width=3: clamp acc+(2*input-1)
// to at least -4, emit at one, then subtract the emitted bit. Active-low reset
// clears acc.
// There is no high clamp: acc <= 2 always holds here, so acc_sum = acc + step
// <= 3 can never exceed the width-3 upper bound. By induction, acc is 0 <= 2
// after reset; given acc <= 2, the step is +1 or -1, so acc_sum <= 3. If the
// clamped sum is >= 1 the emit subtracts exactly 1, leaving <= 2; otherwise the
// state is the clamped sum, which is <= 0 <= 2. Only the sibling uni2bi helper,
// whose steps are all positive, climbs onto an upper bound, and there the fire
// threshold is 2 rather than 1.


module div_iscb_bi2uni (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    // 4-bit signed accumulator holds [-4, 2] with slack for the +/-1 step.
    localparam signed [3:0] ACC_MIN = -4'sd4;

    reg signed [3:0] acc_q;

    // acc += 2*in - 1, clamped.
    wire signed [3:0] step    = i_input ? 4'sd1 : -4'sd1;
    wire signed [3:0] acc_sum = acc_q + step;
    wire signed [3:0] acc_clmp = (acc_sum < ACC_MIN) ? ACC_MIN : acc_sum;

    wire out = (acc_clmp >= 4'sd1);    // acc >= 1
    assign o_out = out;

    // acc -= out (carry-out removed); stays within [-4, 2].
    wire signed [3:0] acc_next = out ? (acc_clmp - 4'sd1) : acc_clmp;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc_q <= 4'sd0;
        else
            acc_q <= acc_next;
    end
endmodule
`default_nettype wire
