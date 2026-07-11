`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// gt_rc -- "greater than" of two rate-coded spike streams via sync_skewed.
//
// Gate-level model of napl.operation.gt_rc (config-free, no polarity branch).
// State: a 1-bit result register (o_out, reset to 1) and the 2-bit skew counter
// of sync_skewed (width=2, cnt in 0..3, reset to 0). One posedge i_clk == one
// Python forward() timestep; i_rst_n (active low) == Python reset().
//
// Per cycle (a=i_in_0, b=i_in_1), reproducing sync_skewed(width=2) then gt_rc:
//   diff        = a ^ b                         // sync's input_01_10
//   sync_0      = diff ? (a ? (cnt==3) : (cnt!=0)) : a   // skewed input_0
//   sync_1      = b                             // input_1 passthrough
//   d_enable    = sync_0 ^ sync_1
//   dff_next    = d_enable ? sync_0 : dff       // result register
//   cnt_next    = clamp(cnt + (diff ? (a ? +1 : -1) : 0), 0, 3)
// o_out is the registered dff read BEFORE its update, so latency is 1 cycle.
//==============================================================================
module gt_rc (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in_0,
    input  wire i_in_1,
    output wire o_out
);
    reg       dff;        // result register, reset to 1
    reg [1:0] cnt;        // sync_skewed skew counter, 0..3, reset to 0

    wire       a    = i_in_0;
    wire       b    = i_in_1;
    wire       diff = a ^ b;

    wire cnt_not_min = (cnt != 2'd0);
    wire cnt_not_max = (cnt != 2'd3);

    // sync_skewed output_1 (a's skewed stream): see header derivation.
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

    assign o_out = dff;
endmodule
`default_nettype wire
