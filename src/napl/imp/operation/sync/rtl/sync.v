`timescale 1ns/1ps
`default_nettype none
// Synchronizer equivalent to napl.sim.operation.sync; pp_delay=0.
// A signed saved-bit counter in [-DEPTH, DEPTH] pairs the unpaired bits of the
// two streams: positive holds saved i_input_0 bits, negative saved i_input_1
// bits. Outputs are combinational in the pre-update counter. The circuit is
// identical for both polarities, so there is no polarity variant.
// Active-low reset clears the counter.


module sync #(
    parameter integer DEPTH = 1   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,       // one posedge == one Python forward() timestep
    input  wire i_rst_n,     // active-low; maps to Python reset() (cnt = 0)
    input  wire i_input_0,   // first input spike stream
    input  wire i_input_1,   // second input spike stream
    output wire o_output_0,  // first synchronized spike stream
    output wire o_output_1   // second synchronized spike stream
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
    localparam [CNT_W-1:0] CNT_ONE = {{(CNT_W-1){1'b0}}, 1'b1};

    reg signed [CNT_W-1:0] cnt;

    // The counter moves by one step per cycle and is held at each rail, so it
    // stays inside [-DEPTH, DEPTH] and equality tests the rail.
    wire agree   = i_input_0 & i_input_1;
    wire is_1_0  = i_input_0 & ~i_input_1;
    wire is_0_1  = ~i_input_0 & i_input_1;
    wire saved_1 = cnt[CNT_W-1];
    wire saved_0 = ~cnt[CNT_W-1] & (|cnt);
    wire full_0  = (cnt == CNT_MAX[CNT_W-1:0]);
    wire full_1  = (cnt == CNT_MIN[CNT_W-1:0]);

    // An unpaired i_input_0 bit releases a saved i_input_1 bit on both outputs,
    // otherwise it is saved silently, and it passes through only once the
    // i_input_0 save slots are full. The i_input_1 side mirrors that.
    assign o_output_0 = agree | (is_1_0 & (saved_1 | full_0)) | (is_0_1 & saved_0);
    assign o_output_1 = agree | (is_1_0 & saved_1) | (is_0_1 & (saved_0 | full_1));

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cnt <= {CNT_W{1'b0}};
        else if (is_1_0 && !full_0)
            cnt <= cnt + $signed(CNT_ONE);
        else if (is_0_1 && !full_1)
            cnt <= cnt - $signed(CNT_ONE);
    end
endmodule
`default_nettype wire
