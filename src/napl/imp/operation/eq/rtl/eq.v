`timescale 1ns/1ps
`default_nettype none
// Equality comparator eq: one unsigned WIDTH-bit saturating counter starting at
// HALF = 2**(WIDTH-1). Input pair 10 counts up, 01 counts down, 00 and 11 hold,
// and the counter saturates at 0 and 2**WIDTH-1. The output is
// o_output = |cnt_next - HALF| <= TOLERANCE, taken on the post-update counter,
// so it is combinational (pp_delay = 0). Polarity does not change the circuit.
// One posedge == one Python timestep; active-low i_rst_n reloads HALF.


module eq #(
    parameter integer WIDTH     = 3,  // inherited from config['width']; tb overrides via `GEN_WIDTH
    parameter integer TOLERANCE = 1   // inherited from config['tolerance']; tb overrides via `GEN_TOLERANCE
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);
    localparam [WIDTH-1:0] ZERO = {WIDTH{1'b0}};
    localparam [WIDTH-1:0] MAX  = {WIDTH{1'b1}};        // 2**WIDTH - 1
    localparam [WIDTH-1:0] HALF = (1 << (WIDTH - 1));   // 2**(WIDTH-1)
    // Band half-width, sized to the WIDTH+1-bit compare below.
    localparam [WIDTH:0]   TOL  = TOLERANCE;

    reg [WIDTH-1:0] cnt;

    wire up   =  i_input_0 & ~i_input_1;
    wire down = ~i_input_0 &  i_input_1;

    // Saturating up/down step; a 00 or 11 pair leaves the counter unchanged.
    wire [WIDTH-1:0] cnt_next =
        up   ? ((cnt == MAX)  ? MAX  : cnt + 1'b1) :
        down ? ((cnt == ZERO) ? ZERO : cnt - 1'b1) : cnt;

    wire [WIDTH-1:0] band = (cnt_next >= HALF) ? (cnt_next - HALF) : (HALF - cnt_next);

    assign o_output = ({1'b0, band} <= TOL);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cnt <= HALF;
        else
            cnt <= cnt_next;
    end
endmodule
`default_nettype wire
