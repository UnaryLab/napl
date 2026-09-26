`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.add_scale with an ENTRY-lane spike input.
// The accumulator adds the partial sum, clamps to at most 2^(WIDTH-1)-1, fires at
// or above SCALE, and subtracts SCALE on a fire. Generated parameters mirror Python.
// There is no negative clamp: acc >= 0 always holds here. By induction, acc is 0
// after reset; given acc >= 0, the unipolar offset is 0 and every addend is
// non-negative, so the sum is >= acc >= 0, and a fire needs sum >= SCALE and
// subtracts exactly SCALE, leaving the state >= 0 again. Only the bipolar
// variant, whose nonzero offset is subtracted every cycle, can drive a low clamp,
// and it carries one.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the accumulator to match reset().
// The sizing guard is the bipolar variant's: 2**WIDTH + 3*ENTRY + 1 must stay at
// or below 2**31-1, which also caps WIDTH at 30 on its own, and mapping.yaml
// carries the same restriction. The same guard also rejects WIDTH below 2, where
// the WIDTH-1 bit accumulator would be empty and Python admits no SCALE, since it
// requires SCALE to be at most 2^(WIDTH-1)-1.


module add_scale_unipolar #(
    parameter integer SCALE = 8,     // inherited from config['scale']; tb overrides via `GEN_SCALE
    parameter integer WIDTH = 20,    // inherited from config['intwidth']; tb overrides via `GEN_WIDTH
    parameter integer ENTRY = 8      // # addends (reduction dim);   tb overrides via `GEN_ENTRY
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire [ENTRY-1:0]      i_input,
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
    // ENTRY <= SCALE: a state of at most SCALE-1 plus at most ENTRY stays below
    // 2*SCALE, so after the fire or without one acc stays in [0, SCALE-1], which
    // clog2(SCALE) unsigned bits hold; SCALE 1 keeps acc at 0 in one bit.
    // ENTRY > SCALE: acc climbs onto the clamp ACC_HI, so it spans [0, ACC_HI],
    // which WIDTH-1 unsigned bits hold.
    localparam integer NARROW = (ENTRY <= SCALE) ? 1 : 0;
    localparam integer ACC_W = (NARROW == 0) ? WIDTH - 1
                             : (SCALE > 1) ? clog2(SCALE) : 1;
    // The pre-clamp sum reaches ACC_HI + ENTRY, and SCALE <= ACC_HI shares the bus.
    localparam integer SUM_W = clog2((2 ** (WIDTH - 1)) + ENTRY);

    // Largest 3*ENTRY+1 the 32-bit signed constants leave room for, written as
    // 2*(2**30 - 2**(WIDTH-1)) - 1 == 2**31-1 - 2**WIDTH so the bound itself
    // never overflows.
    localparam integer SUM_MAX = 2 * ((2 ** 30) - (2 ** (WIDTH - 1))) - 1;

    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when WIDTH or ENTRY leaves the 32-bit range above, and when
    // WIDTH is below 2; the module name records only the 32-bit terms.
    generate
        if (WIDTH < 2 || WIDTH > 30 || (3 * ENTRY + 1) > SUM_MAX) begin : g_bad_sizing
            ERROR_add_scale_WIDTH_and_ENTRY_overflow_32_bit_constants u_bad ();
        end
    endgenerate

    reg [ACC_W-1:0] acc;

    wire [COUNT_W-1:0] partial_count [0:ENTRY];
    assign partial_count[0] = {COUNT_W{1'b0}};

    genvar lane;
    generate
        for (lane = 0; lane < ENTRY; lane = lane + 1) begin : g_count
            // Padded by a full COUNT_W zeros and sliced so no replication is empty.
            /* verilator lint_off UNUSEDSIGNAL */
            wire [COUNT_W:0] lane_pad = {{COUNT_W{1'b0}}, i_input[lane]};
            /* verilator lint_on UNUSEDSIGNAL */
            assign partial_count[lane+1] = partial_count[lane] + lane_pad[COUNT_W-1:0];
        end
    endgenerate

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    wire [SUM_W-1:0] acc_ext   = {{(SUM_W-ACC_W){1'b0}}, acc};
    // SUM_W can equal COUNT_W (WIDTH 3 with ENTRY 4), so the count is padded by a
    // full SUM_W zeros and the low SUM_W bits are kept, which never needs an empty
    // replication.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [SUM_W+COUNT_W-1:0] in_pad = {{SUM_W{1'b0}}, partial_count[ENTRY]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [SUM_W-1:0] in_ext    = in_pad[SUM_W-1:0];
    wire [SUM_W-1:0] scale_ext = SCALE[SUM_W-1:0];
    wire [SUM_W-1:0] sum  = acc_ext + in_ext;
    wire [SUM_W-1:0] clmp = (sum > ACC_HI[SUM_W-1:0]) ? ACC_HI[SUM_W-1:0] : sum;
    wire             fire = (clmp >= scale_ext);
    // clmp is at most ACC_HI so it fits ACC_W; nxt = fired? clmp-SCALE : clmp stays
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
