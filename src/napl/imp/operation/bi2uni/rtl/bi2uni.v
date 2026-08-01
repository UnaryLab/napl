`timescale 1ns/1ps
`default_nettype none
// Bipolar-to-unipolar equivalent of napl.sim.operation.bi2uni.
// Each cycle clamps acc+(2*input-1) to the WIDTH-derived range, emits when the
// updated value is at least one, then subtracts the emitted bit.
// Output is combinational (pp_delay=0); active-low reset clears acc.
module bi2uni #(
    parameter integer WIDTH = 2   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,   // active-low reset -> Python reset(): acc = 0
    input  wire i_input,      // input spike (bipolar stream)
    output wire o_out      // output spike (unipolar stream)
);
    // (WIDTH+1)-bit signed datapath: holds acc in [ACC_MIN, ACC_MAX] and the
    // pre-clamp sum in [ACC_MIN-1, ACC_MAX+1].
    localparam integer DW = WIDTH + 1;

    localparam signed [DW-1:0] ACC_MAX =  (1 <<< (WIDTH-1)) - 1;
    localparam signed [DW-1:0] ACC_MIN = -(1 <<< (WIDTH-1));
    localparam signed [DW-1:0] ONE      =  1;

    reg signed [DW-1:0] acc;

    wire signed [DW-1:0] addend  = i_input ? ONE : -ONE;
    wire signed [DW-1:0] sum     = acc + addend;
    wire signed [DW-1:0] clamped = (sum > ACC_MAX) ? ACC_MAX :
                                   (sum < ACC_MIN) ? ACC_MIN : sum;

    assign o_out = (clamped >= ONE);

    wire signed [DW-1:0] acc_nxt = o_out ? (clamped - ONE) : clamped;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {DW{1'b0}};
        else
            acc <= acc_nxt;
    end
endmodule
`default_nettype wire
