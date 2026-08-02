`timescale 1ns/1ps
`default_nettype none
// Unipolar-to-bipolar equivalent of napl.sim.operation.uni2bi.
// Each cycle clamps acc+input+1 to the WIDTH-derived range, emits at two, then
// subtracts twice the emitted bit. Output is combinational (pp_delay=0).
// Active-low reset clears acc.


module uni2bi #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,   // active-low reset -> Python reset(): acc = 0
    input  wire i_input,      // input spike (unipolar stream)
    output wire o_out      // output spike (bipolar stream)
);
    // acc range [ACC_MIN, ACC_MAX]; a (WIDTH+1)-bit signed bus also covers the
    // pre-clamp sum acc+addend (addend in {1,2}).
    localparam signed [WIDTH:0] ACC_MAX =  (1 << (WIDTH-1)) - 1;
    localparam signed [WIDTH:0] ACC_MIN = -(1 << (WIDTH-1));

    reg signed [WIDTH:0] acc;

    wire signed [WIDTH:0] addend  = i_input ? 2 : 1;
    wire signed [WIDTH:0] sum     = acc + addend;
    wire signed [WIDTH:0] clamped = (sum > ACC_MAX) ? ACC_MAX :
                                    (sum < ACC_MIN) ? ACC_MIN : sum;

    assign o_out = (clamped >= 2);

    wire signed [WIDTH:0] acc_nxt = o_out ? (clamped - 2) : clamped;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {(WIDTH+1){1'b0}};
        else
            acc <= acc_nxt;
    end
endmodule
`default_nettype wire
