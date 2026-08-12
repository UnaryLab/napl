`timescale 1ns/1ps
`default_nettype none
// Saturating running-count accumulator equivalent to napl.sim.operation.counter.
// The register holds the running spike count; each posedge adds this cycle's input
// and saturates at MAX = 2**WIDTH - 1. The saturation-flag output is combinational
// on the just-updated count (pp_delay=0), so a unipolar input reaches the output in
// its arrival cycle. Active-low reset clears the count to 0 (the model's post-reset
// state). WIDTH is inherited from config['width']; the tb overrides via `GEN_WIDTH.


module counter #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,     // one posedge == one Python forward() timestep
    input  wire i_rst_n,   // active-low; maps to Python reset() (count = 0)
    input  wire i_input,   // unipolar 0/1 spike for this timestep
    output wire o_output   // saturation flag: 1 once the count has reached MAX
);
    localparam [WIDTH-1:0] MAX = {WIDTH{1'b1}};   // 2**WIDTH - 1

    reg [WIDTH-1:0] count;

    // The count only ever rises (unipolar input), so it never underflows 0; once at
    // MAX it holds, and below MAX adding one bit cannot overflow WIDTH bits.
    wire at_max = (count == MAX);
    wire [WIDTH-1:0] nxt = at_max ? MAX : (count + {{(WIDTH-1){1'b0}}, i_input});

    // Combinational flag on the just-updated count, matching the model's per-timestep readout.
    assign o_output = (nxt == MAX);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            count <= {WIDTH{1'b0}};
        else
            count <= nxt;
    end
endmodule
`default_nettype wire
