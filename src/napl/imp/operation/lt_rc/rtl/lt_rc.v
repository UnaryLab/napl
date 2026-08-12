`timescale 1ns/1ps
`default_nettype none
// Rate-coded lt_rc equivalent with a width-2 sync_skewed counter.
// o_output is the pre-update result register, so pp_delay=1.
// Each posedge is one Python timestep; active-low reset clears result and cnt.


module lt_rc (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input_0,   // spike stream 0 (the "smaller" operand to sync)
    input  wire i_input_1,   // spike stream 1 (kept unchanged through sync)
    output wire o_output  // registered less-than spike
);
    // sync_skewed state: 2-bit saturating counter, range 0..3.
    reg  [1:0] cnt;
    // lt_rc state: running less-than result.
    reg        dff;

    // ---- combinational sync_skewed datapath (cnt held constant) ----
    wire diff    = i_input_0 ^ i_input_1;            // pair is 01 or 10
    wire not_min = (cnt != 2'd0);
    wire not_max = (cnt != 2'd3);

    // select = not_min - (not_min + not_max) * i_input_0   (small signed integer)
    // sync_0 = i_input_0 + diff * select, which is always a 0/1 spike:
    //   diff==0          -> sync_0 = i_input_0
    //   diff==1, in_0==0 -> sync_0 = not_min       (0 + 1*not_min)
    //   diff==1, in_0==1 -> sync_0 = 1 - not_max   (1 + 1*(-not_max))
    wire sync_0 = diff ? (i_input_0 ? (~not_max) : not_min) : i_input_0;
    wire sync_1 = i_input_1;

    // counter next-state with saturation at the boundaries.
    reg  [1:0] cnt_nxt;
    always @* begin
        cnt_nxt = cnt;
        if (diff) begin
            if (i_input_0) begin
                if (not_max) cnt_nxt = cnt + 2'd1;   // add 1, else hold at 3
            end else begin
                if (not_min) cnt_nxt = cnt - 2'd1;   // sub 1, else hold at 0
            end
        end
    end

    // ---- lt_rc combinational update ----
    wire d_en    = sync_0 ^ sync_1;
    wire dff_nxt = d_en ? sync_1 : dff;

    // o_output is the dff value before this cycle's update (registered output).
    assign o_output = dff;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= 2'd0;
            dff <= 1'b0;
        end else begin
            cnt <= cnt_nxt;
            dff <= dff_nxt;
        end
    end
endmodule
`default_nettype wire
