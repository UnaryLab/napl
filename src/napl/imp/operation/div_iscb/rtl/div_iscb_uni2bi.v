`timescale 1ns/1ps
`default_nettype none
// Fixed-width uni2bi helper matching Python: add (input+1) to acc, emit at two,
// then subtract twice the emitted bit. Active-low reset clears acc.
// There is no clamp at either end: acc is 0 or 1 always. By induction, acc is 0
// after reset; given acc in [0,1], the step is 1 or 2, so acc_sum is in [1,3]. A
// fire needs acc_sum >= 2 and subtracts exactly 2, leaving 0 or 1; a non-fire
// leaves acc_sum = 1. So acc stays in [0,1] and acc_sum stays in [1,3], inside
// the model's fixed [-4, 3] bounds at either end. The sibling bi2uni helper,
// whose step can be -1, is the one that walks onto a bound and carries a clamp.


module div_iscb_uni2bi (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    // 5-bit signed accumulator carries the model's [-4, 3] range.
    reg signed [4:0] acc_q;

    // acc += (in + 1).
    wire signed [4:0] step    = i_input ? 5'sd2 : 5'sd1;
    wire signed [4:0] acc_sum = acc_q + step;

    wire out = (acc_sum >= 5'sd2);    // acc >= 2
    assign o_out = out;

    // acc -= 2*out; stays within [0, 1].
    wire signed [4:0] acc_next = out ? (acc_sum - 5'sd2) : acc_sum;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc_q <= 5'sd0;
        else
            acc_q <= acc_next;
    end
endmodule
`default_nettype wire
