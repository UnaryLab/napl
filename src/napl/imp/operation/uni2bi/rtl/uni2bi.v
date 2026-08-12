`timescale 1ns/1ps
`default_nettype none
// Unipolar-to-bipolar equivalent of napl.sim.operation.uni2bi.
// Each cycle adds input+1 to acc, emits at two, then subtracts twice the emitted
// bit. Output is combinational (pp_delay=0). Active-low reset clears acc.
// There is no clamp at either end: acc is 0 or 1 always. By induction, acc is 0
// after reset; given acc in [0,1], the addend is 1 or 2, so sum is in [1,3]. A
// fire needs sum >= 2 and subtracts exactly 2, leaving 0 or 1; a non-fire leaves
// sum = 1. So acc stays in [0,1] and sum stays in [1,3]. The Python model rejects
// a width whose acc_max = 2^(WIDTH-1)-1 is below the emission threshold 2, so
// every legal WIDTH has acc_max >= 3 >= sum and acc_min = -2^(WIDTH-1) <= 0 <= acc:
// neither clamp arm can ever be taken at any legal size.


module uni2bi #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,   // active-low reset -> Python reset(): acc = 0
    input  wire i_input,      // input spike (unipolar stream)
    output wire o_output   // output spike (bipolar stream)
);
    // acc range [0, 1]; a (WIDTH+1)-bit signed bus also covers the sum
    // acc+addend (addend in {1,2}).
    reg signed [WIDTH:0] acc;

    wire signed [WIDTH:0] addend = i_input ? 2 : 1;
    wire signed [WIDTH:0] sum    = acc + addend;

    assign o_output = (sum >= 2);

    wire signed [WIDTH:0] acc_nxt = o_output ? (sum - 2) : sum;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {(WIDTH+1){1'b0}};
        else
            acc <= acc_nxt;
    end
endmodule
`default_nettype wire
