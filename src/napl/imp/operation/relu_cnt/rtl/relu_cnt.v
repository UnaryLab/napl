`timescale 1ns/1ps
`default_nettype none
// Bipolar relu_cnt equivalent with a WIDTH-bit saturating accumulator.
// Output is combinational (pp_delay=0); each posedge updates acc by the emitted
// spike. Active-low reset loads HALF=2**(WIDTH-1).


module relu_cnt #(
    parameter integer WIDTH = 3  // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,    // input spike (bipolar rate-coded)
    output wire o_output // ReLU output spike
);
    localparam [WIDTH-1:0] MAX  = {WIDTH{1'b1}};        // 2^WIDTH - 1
    localparam [WIDTH-1:0] HALF = (1 << (WIDTH - 1));   // 2^(WIDTH-1)

    reg [WIDTH-1:0] acc;

    wire below_half = (acc < HALF);

    assign o_output = i_input | below_half;

    // Up/down saturating counter: +1 when o_output, -1 otherwise, clamped to [0, MAX].
    wire [WIDTH-1:0] acc_next =
        o_output ? ((acc == MAX) ? MAX : acc + 1'b1)
              : ((acc == {WIDTH{1'b0}}) ? {WIDTH{1'b0}} : acc - 1'b1);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= HALF;
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
