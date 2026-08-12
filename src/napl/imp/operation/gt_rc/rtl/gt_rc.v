`timescale 1ns/1ps
`default_nettype none
// Rate-coded gt_rc equivalent with a width-2 sync_skewed counter.
// o_output is the pre-update result register, so pp_delay=1.
// Each posedge is one Python timestep; active-low reset loads result=1, cnt=0.


module gt_rc (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);
    reg       dff;        // result register, reset to 1
    reg [1:0] cnt;        // sync_skewed skew counter, 0..3, reset to 0

    wire       a    = i_input_0;
    wire       b    = i_input_1;
    wire       diff = a ^ b;

    wire cnt_not_min = (cnt != 2'd0);
    wire cnt_not_max = (cnt != 2'd3);

    // sync_skewed output_0 (a's skewed stream): see header derivation.
    wire sync_0 = diff ? (a ? ~cnt_not_max : cnt_not_min) : a;
    wire sync_1 = b;

    wire d_enable = sync_0 ^ sync_1;
    wire dff_next = d_enable ? sync_0 : dff;

    // cnt += diff ? (a ? +1 : -1) : 0, saturating at 0 and 3.
    reg [1:0] cnt_next;
    always @(*) begin
        cnt_next = cnt;
        if (diff) begin
            if (a) begin
                if (cnt_not_max)
                    cnt_next = cnt + 2'd1;
            end else begin
                if (cnt_not_min)
                    cnt_next = cnt - 2'd1;
            end
        end
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            dff <= 1'b1;
            cnt <= 2'd0;
        end else begin
            dff <= dff_next;
            cnt <= cnt_next;
        end
    end

    assign o_output = dff;
endmodule
`default_nettype wire
