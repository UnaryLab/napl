`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// min_rc -- rate-coded streaming min / argmin via sync_skewed.
//
// Gate-level counterpart of napl.operation.min_rc (operation/compare.py), which
// in turn uses operation/sync.py::sync_skewed with width=2 (cnt in 0..3).
//
// One posedge i_clk == one Python forward() timestep. Active-low i_rst_n maps to
// the Python reset(): cnt <- 0, dff <- 0.
//
// Inputs  i_in_0, i_in_1 : 1-bit spikes (sync_skewed receives them as
//                          input_1=i_in_0, input_2=i_in_1).
// Outputs o_min          : min spike    = dff*i_in_0 + (1-dff)*i_in_1 (PRE-update dff)
//         o_argmin       : argmin spike  = 1 - dff_next               (POST-update dff)
//
// Mirroring the Python forward(): the min uses the dff value held coming into the
// timestep, while the argmin is read AFTER the in-timestep dff update (the model
// returns 1 - self.dff on the line following self.dff.data = ...). Both are thus
// combinational w.r.t. the inputs (delay 0); dff/cnt latch on the clock edge.
//==============================================================================
module min_rc (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in_0,
    input  wire i_in_1,
    output wire o_min,
    output wire o_argmin
);
    // --- registered state (post-reset() == 0) -------------------------------
    reg [1:0] cnt;   // sync_skewed counter, width=2 -> max 3
    reg       dff;   // min_rc argmin/select flop

    localparam [1:0] CNT_MAX = 2'd3;

    // --- sync_skewed combinational datapath ---------------------------------
    // input_01_10 : the two spikes differ (01 or 10)
    wire x01 = i_in_0 ^ i_in_1;

    wire cnt_not_min = (cnt != 2'd0);
    wire cnt_not_max = (cnt != CNT_MAX);

    // select = cnt_not_min - (cnt_not_min + cnt_not_max) * i_in_0
    // sync_0  = i_in_0 + x01 * select   (always 0/1; case-derived below)
    //   x01==0          -> sync_0 = i_in_0
    //   x01==1,i_in_0==1 -> sync_0 = 1 - cnt_not_max  (=cnt_not_max ? 0 : 1)
    //   x01==1,i_in_0==0 -> sync_0 = cnt_not_min
    wire sync_0 = x01 ? (i_in_0 ? ~cnt_not_max : cnt_not_min) : i_in_0;
    wire sync_1 = i_in_1;  // input_2 passes through unchanged

    // --- min_rc combinational datapath --------------------------------------
    wire d_enable = sync_0 ^ sync_1;
    wire and_gate = sync_1 & d_enable;

    // next-state for dff: update to and_gate when d_enable, else hold
    wire dff_next = d_enable ? and_gate : dff;

    // min output uses the ORIGINAL inputs and the pre-update dff;
    // argmin is read after the in-timestep dff update (1 - dff_next)
    assign o_min    = dff ? i_in_0 : i_in_1;
    assign o_argmin = ~dff_next;

    // next-state for cnt: cnt + x01*(2*i_in_0 - 1), saturated to [0, CNT_MAX]
    // only changes when x01 (the inputs differ).
    reg [1:0] cnt_next;
    always @(*) begin
        cnt_next = cnt;
        if (x01) begin
            if (i_in_0) begin
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
