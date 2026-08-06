`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.add_any with an ENTRY-lane spike input.
// A adds 2*partial, clamps to [-2^WIDTH, 2^WIDTH-2], fires at or above 2*SCALE, and
// subtracts 2*SCALE on a fire. Generated parameters mirror Python.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().
// The accumulator bounds and the pre-clamp sum are 32-bit signed elaboration
// constants, so WIDTH and ENTRY are capped jointly: 2**WIDTH + 3*ENTRY + 1 must
// stay at or below 2**31-1, which also caps WIDTH at 30 on its own. The generate
// guard below enforces both terms at elaboration and mapping.yaml carries the
// same restriction.


module add_any_unipolar #(
    parameter integer SCALE = 8,     // inherited from config['scale']; tb overrides via `GEN_SCALE
    parameter integer WIDTH = 20,    // inherited from config['width']; tb overrides via `GEN_WIDTH
    parameter integer ENTRY = 8      // # addends (reduction dim);   tb overrides via `GEN_ENTRY
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire [ENTRY-1:0]      i_input,
    output wire                  o_out
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

    localparam integer TWO_OFS = 0;                  // 2*offset = 0 (unipolar)
    localparam integer TWO_SCL = 2 * SCALE;          // 2*scale
    localparam integer ACC_HI  = (2 ** WIDTH) - 2;   // 2*(2^(WIDTH-1)-1)
    localparam integer ACC_LO  = -(2 ** WIDTH);      // 2*(-2^(WIDTH-1))

    localparam integer COUNT_W = clog2(ENTRY + 1);
    // A = 2*acc; signed reg of width WIDTH+1 covers [-2^WIDTH, 2^WIDTH-1] ⊇ state.
    localparam integer ACC_W = WIDTH + 1;
    // pre-clamp sum |value| <= 2^WIDTH + 2*ENTRY + |TWO_OFS| <= 2^WIDTH + 3*ENTRY;
    // size a signed bus to hold it (clog2 magnitude + sign).
    localparam integer SUM_W = clog2((2 ** WIDTH) + 3 * ENTRY + 1) + 1;

    // Largest 3*ENTRY+1 the 32-bit signed constants leave room for, written as
    // 2*(2**30 - 2**(WIDTH-1)) - 1 == 2**31-1 - 2**WIDTH so the bound itself
    // never overflows.
    localparam integer SUM_MAX = 2 * ((2 ** 30) - (2 ** (WIDTH - 1))) - 1;

    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when WIDTH or ENTRY pushes the constants above out of the
    // 32-bit signed range.
    generate
        if (WIDTH > 30 || (3 * ENTRY + 1) > SUM_MAX) begin : g_bad_sizing
            ERROR_add_any_WIDTH_and_ENTRY_overflow_32_bit_constants u_bad ();
        end
    endgenerate

    reg signed [ACC_W-1:0] acc;

    wire [COUNT_W-1:0] partial_count [0:ENTRY];
    assign partial_count[0] = {COUNT_W{1'b0}};

    genvar lane;
    generate
        for (lane = 0; lane < ENTRY; lane = lane + 1) begin : g_count
            assign partial_count[lane+1] = partial_count[lane]
                + {{(COUNT_W-1){1'b0}}, i_input[lane]};
        end
    endgenerate

    // Signed constants sized to the datapath so every compare/add is signed-vs-signed
    // (slice the 32-bit integer localparam down to SUM_W, sign preserved).
    wire signed [SUM_W-1:0] s_hi  = ACC_HI[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_lo  = ACC_LO[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_scl = TWO_SCL[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_ofs = TWO_OFS[SUM_W-1:0];

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    // 2*partial as a signed value: popcount the spike lanes, then zero-extend.
    wire signed [SUM_W-1:0] in_ext =
        $signed({{(SUM_W-COUNT_W){1'b0}}, partial_count[ENTRY]});
    wire signed [SUM_W-1:0] two_p  = in_ext <<< 1;
    wire signed [SUM_W-1:0] sum   = $signed(acc) + two_p - s_ofs;   // + 2*partial - 2*offset
    wire signed [SUM_W-1:0] clmp  = (sum > s_hi) ? s_hi :
                                    (sum < s_lo) ? s_lo : sum;
    wire                    fire  = (clmp >= s_scl);
    // clmp is in [ACC_LO,ACC_HI] so it fits ACC_W signed; nxt = fired? clmp-TWO_SCL : clmp
    // stays within [ACC_LO, ACC_HI], also ACC_W signed.
    wire signed [ACC_W-1:0] nxt   = fire ? (clmp[ACC_W-1:0] - s_scl[ACC_W-1:0])
                                         : clmp[ACC_W-1:0];

    assign o_out = fire;

    // ---- sequential: advance the accumulator; i_rst_n low maps to reset() (acc=0) ----
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {ACC_W{1'b0}};
        else
            acc <= nxt;
    end
endmodule
`default_nettype wire
