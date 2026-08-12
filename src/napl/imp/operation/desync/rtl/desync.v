`timescale 1ns/1ps
`default_nettype none
// Desynchronizer equivalent to napl.sim.operation.desync; pp_delay=0.
// A signed saved-one counter in [-DEPTH, DEPTH] holds ones withheld from one
// stream, and side_0 selects which stream saves next. The saving side flips
// each time the counter returns to zero, so neither output collects all the
// one-sided ones. Outputs are combinational in the pre-update state. The
// circuit is identical for both polarities, so there is no polarity variant.
// Active-low reset clears the counter and restores the first stream as saver.


module desync #(
    parameter integer DEPTH = 1   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,       // one posedge == one Python forward() timestep
    input  wire i_rst_n,     // active-low; maps to Python reset() (cnt = 0, side = +1)
    input  wire i_input_0,   // first input spike stream
    input  wire i_input_1,   // second input spike stream
    output wire o_output_0,  // first desynchronized spike stream
    output wire o_output_1   // second desynchronized spike stream
);


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    // One sign bit above the magnitude that holds DEPTH.
    localparam integer CNT_W = clog2(DEPTH + 1) + 1;
    localparam integer CNT_MAX = DEPTH;
    localparam integer CNT_MIN = -DEPTH;
    localparam integer CNT_NEG_ONE = -1;
    localparam [CNT_W-1:0] CNT_ONE = {{(CNT_W-1){1'b0}}, 1'b1};

    reg signed [CNT_W-1:0] cnt;
    reg side_0;   // 1 when the first stream saves the next paired one

    wire both_1  = i_input_0 & i_input_1;
    wire both_0  = ~i_input_0 & ~i_input_1;
    wire neg     = cnt[CNT_W-1];
    wire zero    = ~(|cnt);
    // The counter moves by one step per cycle away from zero only while its
    // saving side stays fixed, so it stays inside [-DEPTH, DEPTH] and equality
    // tests the rail on whichever side is in use.
    wire full    = (cnt == CNT_MAX[CNT_W-1:0]) | (cnt == CNT_MIN[CNT_W-1:0]);
    wire save    = both_1 & ~full;
    wire emit    = both_0 & ~zero;
    wire at_one  = (cnt == CNT_ONE) | (cnt == CNT_NEG_ONE[CNT_W-1:0]);

    // A saved one is withheld from its own output and released on that same
    // output later, so each stream keeps its value while the ones move apart.
    assign o_output_0 = (i_input_0 & ~i_input_1) | (both_1 & full)
                      | (save & ~side_0) | (emit & ~neg);
    assign o_output_1 = (i_input_1 & ~i_input_0) | (both_1 & full)
                      | (save & side_0) | (emit & neg);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt    <= {CNT_W{1'b0}};
            side_0 <= 1'b1;
        end else begin
            if (save)
                cnt <= side_0 ? (cnt + $signed(CNT_ONE)) : (cnt - $signed(CNT_ONE));
            else if (emit)
                cnt <= side_0 ? (cnt - $signed(CNT_ONE)) : (cnt + $signed(CNT_ONE));
            if (emit & at_one)
                side_0 <= ~side_0;
        end
    end
endmodule
`default_nettype wire
