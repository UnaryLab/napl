`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.div_scale_dyn: one spike stream divided by the scale
// arriving on i_scale each cycle. The accumulator adds the input spike, fires at
// values at or above the current scale, and subtracts that scale on a fire.
// The RTL accumulator counts whole spikes, so i_scale carries an integer scale and
// SCALE_W, ceil(log2(scale_max + 1)), is the width holding it.
// There is no accumulator clamp and no WIDTH parameter here. The unipolar offset
// is 0, so every addend is non-negative; a carry subtracts the current scale, so
// by induction from the post-reset 0 the stored value stays below scale_max and
// the pre-clamp sum never exceeds it. The Python constructor rejects a scale_max
// above acc_max = 2**(intwidth-1)-1, so that sum is always inside the accumulator
// range and intwidth changes no output. The bipolar variant, whose offset is
// negative, does reach its clamp and carries WIDTH.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().


module div_scale_dyn_unipolar #(
    parameter integer SCALE_W = 2    // ceil(log2(scale_max+1)); tb overrides via `GEN_SCALE_W
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire                  i_input,
    input  wire [SCALE_W-1:0]    i_scale,
    output wire                  o_output
);
    // ---- derived sizing (all magic constants trace to the parameters) ----

    // The scale spans [1, scale_max] and the stored value [0, scale_max-1], so the
    // scale width carries both, and the sum, which reaches scale_max.
    localparam integer ACC_W = SCALE_W;

    reg [ACC_W-1:0] acc;

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire [ACC_W-1:0] sum  = acc + i_input;
    wire             fire = (sum >= i_scale);
    wire [ACC_W-1:0] nxt  = fire ? (sum - i_scale) : sum;

    assign o_output = fire;

    // ---- sequential: advance the accumulator; i_rst_n low maps to reset() (acc=0) ----
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {ACC_W{1'b0}};
        else
            acc <= nxt;
    end
endmodule
`default_nettype wire
