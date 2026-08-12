`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.div_scale: one spike stream divided by SCALE.
// The accumulator adds the input spike, fires at values at or above SCALE, and
// subtracts SCALE on a fire, so the circuit is a divide-by-SCALE spike counter.
// Parameters mirror Python.
// There is no accumulator clamp and no WIDTH parameter here. The unipolar offset
// is 0, so every addend is non-negative; a carry subtracts exactly SCALE, so by
// induction from the post-reset 0 the stored value stays in [0, SCALE-1] and the
// pre-clamp sum never exceeds SCALE. The Python constructor rejects a scale above
// acc_max = 2**(intwidth-1)-1, so that sum is always inside the accumulator range
// and intwidth changes no output. The bipolar variant, whose offset is negative,
// does reach its clamp and carries WIDTH.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().


module div_scale_unipolar #(
    parameter integer SCALE = 3      // inherited from config['scale']; tb overrides via `GEN_SCALE
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire                  i_input,
    output wire                  o_output
);
    // ---- ceil(log2(x)) constant function (Verilog-2001) ----


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    // ---- derived sizing (all magic constants trace to the parameters) ----

    // The stored value spans [0, SCALE-1], so ACC_W bits hold it.
    localparam integer ACC_W = clog2(SCALE + 1);
    // One spare bit carries the sum, which reaches SCALE.
    localparam integer SUM_W = ACC_W + 1;

    reg [ACC_W-1:0] acc;

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire [SUM_W-1:0] sum  = {1'b0, acc} + {{ACC_W{1'b0}}, i_input};
    wire             fire = (sum >= SCALE[SUM_W-1:0]);
    // The sum reaches SCALE at most, so a fire subtracts the whole sum and leaves
    // 0; without a fire the sum is below SCALE and fits the accumulator.
    wire [ACC_W-1:0] nxt  = fire ? {ACC_W{1'b0}} : sum[ACC_W-1:0];

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
