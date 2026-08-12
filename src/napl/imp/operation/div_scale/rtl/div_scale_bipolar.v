`timescale 1ns/1ps
`default_nettype none
// Bipolar napl.sim.operation.div_scale: one spike stream divided by SCALE.
// A holds 2*acc, adds 2*input+(SCALE-1) for the offset (1-SCALE)/2, clamps to at
// most 2^WIDTH-2, fires at values at or above 2*SCALE, and subtracts 2*SCALE on a
// fire. Parameters mirror Python.
// There is no negative clamp: SCALE is at least 1, so the addend 2*input+(SCALE-1)
// is non-negative and a carry subtracts exactly 2*SCALE from a value at or above
// it, leaving the state non-negative again by induction from the post-reset 0.
// The whole datapath is therefore unsigned.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().
// The accumulator bound and the pre-clamp sum are 32-bit elaboration constants, so
// WIDTH is capped at 30. The generate guard below enforces that at elaboration and
// mapping.yaml carries the same restriction.


module div_scale_bipolar #(
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

    localparam integer OFS_ADD = SCALE - 1;          // -2*offset = scale - 1
    localparam integer TWO_SCL = 2 * SCALE;          // 2*scale
    localparam integer ACC_HI  = (2 ** WIDTH) - 2;   // 2*(2^(WIDTH-1)-1)

    // A = 2*acc is non-negative and at most ACC_HI, so WIDTH unsigned bits hold it.
    localparam integer ACC_W = WIDTH;
    // pre-clamp sum <= ACC_HI + 2 + OFS_ADD = 2**WIDTH + SCALE - 1; size a bus to hold it.
    localparam integer SUM_W = clog2((2 ** WIDTH) + SCALE);

    // Elaboration-time guard: an unresolvable module reference makes iverilog fail
    // the build when WIDTH pushes the constants above out of the 32-bit range. The
    // SCALE term of that sum needs no guard of its own: the Python constructor
    // rejects a scale above acc_max = 2**(WIDTH-1)-1, so 2**WIDTH + SCALE stays
    // below 1.5 * 2**WIDTH.
    generate
        if (WIDTH > 30) begin : g_bad_sizing
            ERROR_div_scale_WIDTH_overflows_32_bit_constants u_bad ();
        end
    endgenerate

    reg [ACC_W-1:0] acc;

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire [SUM_W-1:0] acc_ext = {{(SUM_W-ACC_W){1'b0}}, acc};
    wire [SUM_W-1:0] in_ext  = {{(SUM_W-2){1'b0}}, i_input, 1'b0};
    wire [SUM_W-1:0] sum  = acc_ext + in_ext + OFS_ADD[SUM_W-1:0];
    wire [SUM_W-1:0] clmp = (sum > ACC_HI[SUM_W-1:0]) ? ACC_HI[SUM_W-1:0] : sum;
    wire             fire = (clmp >= TWO_SCL[SUM_W-1:0]);
    // clmp is at most ACC_HI so it fits ACC_W; nxt = fired? clmp-TWO_SCL : clmp
    // stays within [0, ACC_HI], also ACC_W.
    wire [ACC_W-1:0] nxt  = fire ? (clmp[ACC_W-1:0] - TWO_SCL[ACC_W-1:0])
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
