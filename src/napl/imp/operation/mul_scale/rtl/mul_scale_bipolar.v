`timescale 1ns/1ps
`default_nettype none
// Bipolar napl.sim.operation.mul_scale: one spike stream multiplied by SCALE.
// The sigma-delta dual of div_scale, with the bipolar centering offset negated:
// div_scale adds one unit per spike, fires at SCALE, and centers with +(SCALE-1)/2;
// mul_scale adds SCALE per spike, fires at one unit, and centers with -(SCALE-1)/2.
// A holds 2*acc so the half-unit offset stays integer: it adds 2*SCALE on an input
// spike, subtracts (SCALE-1) for the offset, clamps to [-(2^WIDTH), 2^WIDTH-2],
// fires at values at or above 2 (one whole unit), and drains 2 on a fire. The mean
// output rate settles at v_out = v_in * SCALE. Parameters mirror Python.
// Unlike div_scale, the offset is subtracted, so on a no-spike cycle the addend is
// negative and A falls below zero; the datapath is therefore SIGNED and carries
// both clamps. Amplification by SCALE > 1 also drives A onto the upper clamp.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().
// The accumulator bounds and the pre-clamp sum are 32-bit signed elaboration
// constants, so WIDTH is capped at 30. The generate guard below enforces that at
// elaboration and mapping.yaml carries the same restriction.


module mul_scale_bipolar #(
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

    localparam integer OFS_SUB  = SCALE - 1;         // 2*offset = scale - 1
    localparam integer SPK_ADD  = 2 * SCALE;         // added to A per input spike
    localparam integer TWO_UNIT = 2;                 // one whole unit in A units (fire/drain)
    localparam integer ACC_HI   = (2 ** WIDTH) - 2;  // 2*(2^(WIDTH-1)-1)
    localparam integer ACC_LO   = -(2 ** WIDTH);     // 2*(-2^(WIDTH-1))

    // A = 2*acc; signed reg of width WIDTH+1 covers [-2^WIDTH, 2^WIDTH-1] ⊇ state.
    localparam integer ACC_W = WIDTH + 1;
    // pre-clamp sum |value| <= 2^WIDTH + SPK_ADD <= 2^WIDTH + 2*SCALE; size a signed
    // bus to hold it (clog2 magnitude + sign). SCALE stays below 2^(WIDTH-1) because
    // the Python constructor caps it at acc_max, so the sum never overflows.
    localparam integer SUM_W = clog2((2 ** WIDTH) + SPK_ADD + 1) + 1;

    // Elaboration-time guard: an unresolvable module reference makes iverilog fail
    // the build when WIDTH pushes 2**WIDTH out of the 32-bit signed range.
    generate
        if (WIDTH > 30) begin : g_bad_sizing
            ERROR_mul_scale_WIDTH_overflows_32_bit_constants u_bad ();
        end
    endgenerate

    reg signed [ACC_W-1:0] acc;

    // Signed constants sized to the datapath so every compare/add is signed-vs-signed
    // (slice the 32-bit integer localparam down to SUM_W, sign preserved).
    wire signed [SUM_W-1:0] s_hi  = ACC_HI[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_lo  = ACC_LO[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_spk = SPK_ADD[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_ofs = OFS_SUB[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_two = TWO_UNIT[SUM_W-1:0];

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire signed [SUM_W-1:0] add  = i_input ? s_spk : {SUM_W{1'b0}};
    wire signed [SUM_W-1:0] sum  = $signed(acc) + add - s_ofs;   // + 2*scale*input - 2*offset
    wire signed [SUM_W-1:0] clmp = (sum > s_hi) ? s_hi :
                                   (sum < s_lo) ? s_lo : sum;
    wire                    fire = (clmp >= s_two);
    // clmp is in [ACC_LO,ACC_HI] so it fits ACC_W signed; nxt = fired? clmp-2 : clmp
    // stays within [ACC_LO, ACC_HI], also ACC_W signed.
    wire signed [ACC_W-1:0] nxt  = fire ? (clmp[ACC_W-1:0] - s_two[ACC_W-1:0])
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
