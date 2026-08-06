`timescale 1ns/1ps
`default_nettype none
// Streaming sigmoid_hard equivalent to add_any(scale=2,width=3) on input+1.
// Each cycle adds input+1 to acc, emits at two, then subtracts two.
// Output is combinational (pp_delay=0); active-low reset clears acc.
// There is no clamp at either end: acc is 0 or 1 always. By induction, acc is 0
// after reset; given acc in [0,1], the addend i_input+1 is 1 or 2, so acc_sum is
// in [1,3]. A fire needs acc_sum >= 2 and subtracts exactly 2, leaving 0 or 1; a
// non-fire leaves acc_sum = 1. So acc stays in [0,1] and acc_sum stays in [1,3],
// inside the model's fixed [-4, 3] bounds at either end: neither clamp arm can
// ever be taken.


module sigmoid_hard (
    input  wire i_clk,    // one posedge per Python forward() timestep
    input  wire i_rst_n,  // active-low reset -> Python reset() (acc <- 0)
    input  wire i_input,     // input spike (0/1)
    output wire o_out     // output spike (0/1)
);
    localparam signed [4:0] SCALE = 5'sd2;

    reg signed [4:0] acc;

    // acc + (i_input + 1): i_input extended to the signed accumulator width.
    // The bus carries the model's [-4, 3] range, of which [0, 3] is reachable.
    wire signed [4:0] acc_sum = acc + $signed({4'b0000, i_input}) + 5'sd1;

    // output spike: combinational in (acc, i_input)
    assign o_out = (acc_sum >= SCALE) ? 1'b1 : 1'b0;

    // accumulator update: subtract scale where output fired. result stays in
    // [0, 1].
    wire signed [4:0] acc_next = o_out ? (acc_sum - SCALE) : acc_sum;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= 5'sd0;
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
