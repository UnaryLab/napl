`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.mul_scale: one spike stream multiplied by SCALE.
// The sigma-delta dual of div_scale: div_scale adds one unit per spike and fires
// at SCALE; mul_scale adds SCALE per spike and fires at one unit. The accumulator
// adds SCALE on an input spike, fires at values at or above one unit, and drains
// one unit on a fire, so the mean output rate settles at p_in * SCALE.
// Parameters mirror Python.
// The unipolar offset is 0, so every addend is non-negative and the stored value
// never goes negative; the datapath is unsigned and carries no lower clamp. But
// SCALE > 1 lets the inflow outpace the one-unit drain, so unlike div_scale the
// upper clamp IS reachable and WIDTH is observable, which is why this module
// carries a WIDTH parameter div_scale_unipolar does not.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().


module mul_scale_unipolar #(
    parameter integer SCALE = 3,     // inherited from config['scale']; tb overrides via `GEN_SCALE
    parameter integer WIDTH = 3      // inherited from config['intwidth']; tb overrides via `GEN_WIDTH
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

    // Largest value the signed accumulator retains, in whole units.
    localparam integer ACC_MAX = (2 ** (WIDTH - 1)) - 1;
    // The stored value spans [0, ACC_MAX], so ACC_W bits hold it.
    localparam integer ACC_W = clog2(ACC_MAX + 1);
    // The pre-clamp sum reaches ACC_MAX + SCALE; size a bus to hold it.
    localparam integer SUM_W = clog2(ACC_MAX + SCALE + 1);

    // Elaboration-time guard: 2**(WIDTH-1) is a 32-bit elaboration constant, so
    // cap WIDTH; mapping.yaml carries the same restriction.
    generate
        if (WIDTH > 30) begin : g_bad_sizing
            ERROR_mul_scale_WIDTH_overflows_32_bit_constants u_bad ();
        end
    endgenerate

    reg [ACC_W-1:0] acc;

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    // One input spike carries SCALE units; there is no drain from the addend, so
    // the sum is acc plus SCALE (or acc unchanged) before the clamp.
    wire [SUM_W-1:0] add  = i_input ? SCALE[SUM_W-1:0] : {SUM_W{1'b0}};
    wire [SUM_W-1:0] sum  = {{(SUM_W-ACC_W){1'b0}}, acc} + add;
    wire [SUM_W-1:0] clmp = (sum > ACC_MAX[SUM_W-1:0]) ? ACC_MAX[SUM_W-1:0] : sum;
    // fire at one whole unit; a fire drains exactly one unit and the clamped sum is
    // at least one, so the difference stays within [0, ACC_MAX], also ACC_W.
    wire             fire = (clmp >= {{(SUM_W-1){1'b0}}, 1'b1});
    wire [ACC_W-1:0] nxt  = fire ? (clmp[ACC_W-1:0] - {{(ACC_W-1){1'b0}}, 1'b1})
                                 : clmp[ACC_W-1:0];

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
