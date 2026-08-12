`timescale 1ns/1ps
`default_nettype none
// Bipolar napl.sim.operation.div_scale_dyn: one spike stream divided by the scale
// arriving on i_scale each cycle. A holds 2*acc, adds 2*input+(scale-1) for the
// offset (1-scale)/2, clamps to at most 2^WIDTH-2, fires at values at or above
// 2*scale, and subtracts 2*scale on a fire. Parameters mirror Python.
// The RTL accumulator counts whole spikes, so i_scale carries an integer scale and
// SCALE_W, ceil(log2(scale_max + 1)), is the width holding it.
// There is no negative clamp: the scale is at least 1, so the addend
// 2*input+(scale-1) is non-negative and a carry subtracts exactly 2*scale from a
// value at or above it, leaving the state non-negative again by induction from the
// post-reset 0. The whole datapath is therefore unsigned.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().
// The accumulator bound is a 32-bit elaboration constant, so WIDTH is capped at 30.
// The generate guard below enforces that at elaboration and mapping.yaml carries
// the same restriction.


module div_scale_dyn_bipolar #(
    parameter integer SCALE_W = 2,   // ceil(log2(scale_max+1)); tb overrides via `GEN_SCALE_W
    parameter integer WIDTH   = 3    // inherited from config['intwidth']; tb overrides via `GEN_WIDTH
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire                  i_input,
    input  wire [SCALE_W-1:0]    i_scale,
    output wire                  o_output
);
    // ---- derived sizing (all magic constants trace to the parameters) ----

    localparam integer ACC_HI = (2 ** WIDTH) - 2;   // 2*(2^(WIDTH-1)-1)

    // A = 2*acc is non-negative and at most ACC_HI, so WIDTH unsigned bits hold it.
    localparam integer ACC_W = WIDTH;
    // pre-clamp sum <= ACC_HI + 2 + (scale-1) < 2**WIDTH + 2**SCALE_W, so one bit
    // above the wider of the two carries it.
    localparam integer SUM_W = ((WIDTH > SCALE_W) ? WIDTH : SCALE_W) + 1;

    // Elaboration-time guard: an unresolvable module reference makes iverilog fail
    // the build when WIDTH pushes ACC_HI out of the 32-bit range.
    generate
        if (WIDTH > 30) begin : g_bad_sizing
            ERROR_div_scale_dyn_WIDTH_overflows_32_bit_constants u_bad ();
        end
    endgenerate

    reg [ACC_W-1:0] acc;

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire [SUM_W-1:0] acc_ext   = {{(SUM_W-ACC_W){1'b0}}, acc};
    wire [SUM_W-1:0] in_ext    = {{(SUM_W-2){1'b0}}, i_input, 1'b0};
    wire [SUM_W-1:0] scale_ext = {{(SUM_W-SCALE_W){1'b0}}, i_scale};
    wire [SUM_W-1:0] two_scl   = scale_ext << 1;                  // 2*scale
    wire [SUM_W-1:0] sum  = acc_ext + in_ext + scale_ext - {{(SUM_W-1){1'b0}}, 1'b1};
    wire [SUM_W-1:0] clmp = (sum > ACC_HI[SUM_W-1:0]) ? ACC_HI[SUM_W-1:0] : sum;
    wire             fire = (clmp >= two_scl);
    // clmp is at most ACC_HI so it fits ACC_W; nxt = fired? clmp-two_scl : clmp
    // stays within [0, ACC_HI], also ACC_W.
    wire [ACC_W-1:0] nxt  = fire ? (clmp[ACC_W-1:0] - two_scl[ACC_W-1:0])
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
