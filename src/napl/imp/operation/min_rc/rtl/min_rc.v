`timescale 1ns/1ps
`default_nettype none
// Rate-coded min_rc equivalent with a width-2 sync_skewed counter.
// o_output uses the pre-update arg register; o_index inverts its next value.
// Both are combinational (pp_delay=0). Active-low reset clears arg and cnt.


module min_rc (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output,
    output wire o_index
);
    // --- registered state (post-reset() == 0) -------------------------------
    reg [1:0] cnt;   // sync_skewed counter, width=2 -> max 3
    reg       dff;   // min_rc argmin/select flop

    localparam [1:0] CNT_MAX = 2'd3;

    // --- sync_skewed combinational datapath ---------------------------------
    // input_01_10 : the two spikes differ (01 or 10)
    wire x01 = i_input_0 ^ i_input_1;

    wire cnt_not_min = (cnt != 2'd0);
    wire cnt_not_max = (cnt != CNT_MAX);

    // select = cnt_not_min - (cnt_not_min + cnt_not_max) * i_input_0
    // sync_0  = i_input_0 + x01 * select   (always 0/1; case-derived below)
    //   x01==0          -> sync_0 = i_input_0
    //   x01==1,i_input_0==1 -> sync_0 = 1 - cnt_not_max  (=cnt_not_max ? 0 : 1)
    //   x01==1,i_input_0==0 -> sync_0 = cnt_not_min
    wire sync_0 = x01 ? (i_input_0 ? ~cnt_not_max : cnt_not_min) : i_input_0;
    wire sync_1 = i_input_1;  // i_input_1 passes through unchanged

    // --- min_rc combinational datapath --------------------------------------
    wire d_enable = sync_0 ^ sync_1;
    wire and_gate = sync_1 & d_enable;

    // next-state for dff: update to and_gate when d_enable, else hold
    wire dff_next = d_enable ? and_gate : dff;

    // min output uses the ORIGINAL inputs and the pre-update dff;
    // argmin is read after the in-timestep dff update (1 - dff_next)
    assign o_output = dff ? i_input_0 : i_input_1;
    assign o_index = ~dff_next;

    // next-state for cnt: cnt + x01*(2*i_input_0 - 1), saturated to [0, CNT_MAX]
    // only changes when x01 (the inputs differ).
    reg [1:0] cnt_next;
    always @(*) begin
        cnt_next = cnt;
        if (x01) begin
            if (i_input_0) begin
                if (cnt_not_max) cnt_next = cnt + 2'd1;  // saturate at CNT_MAX
            end else begin
                if (cnt_not_min) cnt_next = cnt - 2'd1;  // saturate at 0
            end
        end
    end

    // --- sequential update ---------------------------------------------------
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
