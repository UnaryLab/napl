`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.add_scale_dyn with an ENTRY-lane spike input and the
// carry scale arriving on i_scale each cycle. The accumulator adds the partial sum,
// clamps to at most 2^(WIDTH-1)-1, fires at values at or above the current scale,
// and subtracts that scale on a fire. Parameters mirror Python.
// The RTL accumulator counts whole spikes, so i_scale carries an integer scale and
// SCALE_W, ceil(log2(scale_max + 1)), is the width holding it.
// There is no negative clamp: acc >= 0 always holds here. By induction, acc is 0
// after reset; given acc >= 0, the unipolar offset is 0 and every addend is
// non-negative, so the sum is >= acc >= 0, and a fire needs sum >= scale and
// subtracts exactly scale, leaving the state >= 0 again. Only the bipolar variant,
// whose nonzero offset is subtracted every cycle, can drive a low clamp, and it
// carries one.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().
// The accumulator bound and the pre-clamp sum are 32-bit elaboration constants, so
// WIDTH, SCALE_W, and ENTRY are capped jointly: WIDTH stays at 30 or below, SCALE_W
// stays below WIDTH (which the Python guard scale_max <= 2**(intwidth-1)-1 already
// requires), and 3*ENTRY+1 must fit the room those leave. The generate guard below
// enforces all three at elaboration and mapping.yaml carries the same restriction.


module add_scale_dyn_unipolar #(
    parameter integer SCALE_W = 2,   // ceil(log2(scale_max+1)); tb overrides via `GEN_SCALE_W
    parameter integer WIDTH   = 6,   // inherited from config['intwidth']; tb overrides via `GEN_WIDTH
    parameter integer ENTRY   = 8    // # addends (reduction dim);  tb overrides via `GEN_ENTRY
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire [ENTRY-1:0]      i_input,
    input  wire [SCALE_W-1:0]    i_scale,
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

    localparam integer ACC_HI = (2 ** (WIDTH - 1)) - 1;

    localparam integer COUNT_W = clog2(ENTRY + 1);
    // acc spans [0, ACC_HI], which WIDTH-1 unsigned bits hold.
    localparam integer ACC_W = WIDTH - 1;
    // pre-clamp sum <= ACC_HI + ENTRY, and the compare against the scale reads the
    // same bus, so one spare bit above all three terms sizes it.
    localparam integer SUM_W = clog2((2 ** (WIDTH - 1)) + ENTRY + (2 ** SCALE_W) + 1) + 1;

    // Largest 3*ENTRY+1 the 32-bit constants leave room for, written as
    // 2*(2**30 - 2**(WIDTH-1) - 2**(WIDTH-2)) - 1 == 2**31-1 - 2**WIDTH - 2**(WIDTH-1)
    // so the bound itself never overflows.
    localparam integer SUM_MAX =
        2 * ((2 ** 30) - (2 ** (WIDTH - 1)) - (2 ** (WIDTH - 2))) - 1;

    // Elaboration-time guard: an unresolvable module reference makes iverilog fail
    // the build when the sizing pushes the constants above out of the 32-bit range.
    generate
        if (WIDTH < 2 || WIDTH > 30 || SCALE_W >= WIDTH
            || (3 * ENTRY + 1) > SUM_MAX) begin : g_bad_sizing
            ERROR_add_scale_dyn_WIDTH_SCALE_W_and_ENTRY_overflow_32_bit_constants u_bad ();
        end
    endgenerate

    reg [ACC_W-1:0] acc;

    wire [COUNT_W-1:0] partial_count [0:ENTRY];
    assign partial_count[0] = {COUNT_W{1'b0}};

    genvar lane;
    generate
        for (lane = 0; lane < ENTRY; lane = lane + 1) begin : g_count
            assign partial_count[lane+1] = partial_count[lane]
                + {{(COUNT_W-1){1'b0}}, i_input[lane]};
        end
    endgenerate

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire [SUM_W-1:0] acc_ext   = {{(SUM_W-ACC_W){1'b0}}, acc};
    wire [SUM_W-1:0] in_ext    = {{(SUM_W-COUNT_W){1'b0}}, partial_count[ENTRY]};
    wire [SUM_W-1:0] scale_ext = {{(SUM_W-SCALE_W){1'b0}}, i_scale};
    wire [SUM_W-1:0] sum  = acc_ext + in_ext;
    wire [SUM_W-1:0] clmp = (sum > ACC_HI[SUM_W-1:0]) ? ACC_HI[SUM_W-1:0] : sum;
    wire             fire = (clmp >= scale_ext);
    // clmp is at most ACC_HI so it fits ACC_W; nxt = fired? clmp-scale : clmp stays
    // within [0, ACC_HI], also ACC_W.
    wire [ACC_W-1:0] nxt  = fire ? (clmp[ACC_W-1:0] - scale_ext[ACC_W-1:0])
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
