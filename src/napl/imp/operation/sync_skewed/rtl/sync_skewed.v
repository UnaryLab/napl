`timescale 1ns/1ps
`default_nettype none
// sync_skewed equivalent for stream_1 rate <= stream_2 rate. A WIDTH-bit
// saturating counter retimes stream_1; stream_2 passes through.
// Outputs use the pre-update counter (pp_delay=0). Active-low reset clears cnt.
module sync_skewed #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (cnt = 0)
    input  wire i_input_1,   // input spike stream 1 (lower rate)
    input  wire i_input_2,   // input spike stream 2 (higher rate, passes through)
    output wire o_out_1,  // re-timed stream 1
    output wire o_out_2   // stream 2, unchanged
);
    localparam integer CNT_MAX = (1 << WIDTH) - 1;

    reg [WIDTH-1:0] cnt;

    wire diff        = i_input_1 ^ i_input_2;          // 01 or 10
    wire cnt_not_min = (cnt != 0);
    wire cnt_not_max = (cnt != CNT_MAX[WIDTH-1:0]);

    // o_out_1: on 00/11 pass i_input_1 through; on a push emit only when saturated
    // high, on a pop emit only when not empty.
    assign o_out_1 = diff ? (i_input_1 ? ~cnt_not_max : cnt_not_min) : i_input_1;
    assign o_out_2 = i_input_2;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= {WIDTH{1'b0}};
        end else if (diff) begin
            if (i_input_1) begin
                if (cnt_not_max) cnt <= cnt + 1'b1;   // push, saturating
            end else begin
                if (cnt_not_min) cnt <= cnt - 1'b1;   // pop, saturating
            end
        end
    end
endmodule
`default_nettype wire
