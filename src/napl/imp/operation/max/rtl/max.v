`timescale 1ns/1ps
`default_nettype none

// Rate-coded max equivalent with a width-2 sync_skewed counter.
// o_output uses the pre-update arg register; o_index uses its next value.
// Both are combinational (pp_delay=0). Active-low reset clears arg and cnt.

module max (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output,
    output wire o_index
);
    localparam [1:0] CNT_MAX = 2'd3;

    reg [1:0] cnt;
    reg       dff;

    // sync_skewed(input_0=i_input_0, input_1=i_input_1).
    wire input_01_10 = i_input_0 ^ i_input_1;              // (i_input_0 + i_input_1) == 1
    wire cnt_not_min = (cnt != 2'd0);
    wire cnt_not_max = (cnt != CNT_MAX);
    // On 00/11 i_input_0 passes through; on a push emit only when saturated high,
    // on a pop emit only when not empty.
    wire sync_0 = input_01_10 ? (i_input_0 ? ~cnt_not_max : cnt_not_min) : i_input_0;
    wire sync_1 = i_input_1;

    // cnt += input_01_10 ? (i_input_0 ? +1 : -1) : 0, saturating at 0 and CNT_MAX.
    reg [1:0] cnt_next;
    always @(*) begin
        cnt_next = cnt;
        if (input_01_10) begin
            if (i_input_0) begin
                if (cnt_not_max) cnt_next = cnt + 2'd1;
            end else begin
                if (cnt_not_min) cnt_next = cnt - 2'd1;
            end
        end
    end

    wire d_enable  = sync_0 ^ sync_1;
    wire dff_next  = d_enable ? sync_1 : dff;

    assign o_output = dff ? i_input_1 : i_input_0;   // pre-update dff
    assign o_index = dff_next;                       // post-update dff

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= 2'd0;
            dff <= 1'b0;
        end else begin
            cnt <= cnt_next;
            dff <= dff_next;
        end
    end
endmodule

`default_nettype wire
