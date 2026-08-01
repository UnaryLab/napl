`timescale 1ns/1ps
`default_nettype none
// Streaming sigmoid_hard equivalent to add_any(scale=2,width=3) on input+1.
// Each cycle clamps acc+input+1 to [-4,3], emits at two, then subtracts two.
// Output is combinational (pp_delay=0); active-low reset clears acc.
module sigmoid_hard (
    input  wire i_clk,    // one posedge per Python forward() timestep
    input  wire i_rst_n,  // active-low reset -> Python reset() (acc <- 0)
    input  wire i_input,     // input spike (0/1)
    output wire o_out     // output spike (0/1)
);
    localparam signed [4:0] ACC_MAX = 5'sd3;
    localparam signed [4:0] ACC_MIN = -5'sd4;
    localparam signed [4:0] SCALE   = 5'sd2;

    reg signed [4:0] acc;

    // acc + (i_input + 1): i_input extended to the signed accumulator width.
    // worst case 3 + 2 = 5, so the running sum is computed at 5 signed bits.
    wire signed [4:0] raw_sum = acc + $signed({4'b0000, i_input}) + 5'sd1;

    // clamp(raw_sum, ACC_MIN, ACC_MAX), all at 5-bit signed width
    wire signed [4:0] acc_sum =
        (raw_sum > ACC_MAX) ? ACC_MAX :
        (raw_sum < ACC_MIN) ? ACC_MIN :
                              raw_sum;

    // output spike: combinational in (acc, i_input)
    assign o_out = (acc_sum >= SCALE) ? 1'b1 : 1'b0;

    // accumulator update: subtract scale where output fired. result stays in
    // [-4, 3], so the low 4 signed bits hold it exactly.
    wire signed [4:0] acc_next = o_out ? (acc_sum - SCALE) : acc_sum;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= 5'sd0;
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
