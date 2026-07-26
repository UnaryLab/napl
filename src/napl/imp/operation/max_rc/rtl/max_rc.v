`timescale 1ns/1ps
`default_nettype none

// max_rc: streaming max + argmax of two rate-coded spike streams via sync_skewed.
// One forward() timestep == one posedge i_clk. Outputs are combinational from the
// current inputs and registered state (sync counter + argmax dff); the registers
// hold next-state and feed back, so pp_delay = 0.
//
// State (post-reset()):
//   cnt = 0   (sync_skewed width-2 saturating counter, range 0..3)
//   dff = 0   (argmax: 0 => input_0 is max, 1 => input_1 is max)
//
// Matches napl.sim.operation.max_rc forward(): o_max uses the OLD dff, o_arg is the
// NEW (post-update) dff, both available the same cycle.
// Verify from src/napl/imp with: make test OP=max_rc
module max_rc (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_max,
    output wire o_arg
);
    localparam [1:0] CNT_MAX = 2'd3;

    reg [1:0] cnt;
    reg       dff;

    // ---- sync_skewed(input_1=i_input_0, input_2=i_input_1) -> sync_0, i_input_1 ----
    wire input_01_10 = i_input_0 ^ i_input_1;              // (i_input_0 + i_input_1) == 1
    wire cnt_not_min = (cnt != 2'd0);
    wire cnt_not_max = (cnt != CNT_MAX);
    // select = cnt_not_min - (cnt_not_min + cnt_not_max) * i_input_0, in {-1,0,1}
    // (cnt_not_min + cnt_not_max) is an arithmetic sum in {0,1,2}, not a concat.
    wire [1:0] not_sum = {1'b0, cnt_not_min} + {1'b0, cnt_not_max};
    wire signed [2:0] select =
        $signed({2'b00, cnt_not_min})
        - (i_input_0 ? $signed({1'b0, not_sum}) : 3'sd0);
    // sync_0 = i_input_0 + input_01_10 * select, provably in {0,1}; low bit is the
    // spike. Upper bits of the sum are always 0 here (intentionally unused).
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [2:0] sync_0_sum = $signed({2'b00, i_input_0})
        + (input_01_10 ? select : 3'sd0);
    /* verilator lint_on UNUSEDSIGNAL */
    wire sync_0 = sync_0_sum[0];
    wire sync_1 = i_input_1;

    // cnt_next = clamp(cnt + input_01_10 * (2*i_input_0 - 1), 0, CNT_MAX)
    wire signed [3:0] cnt_step  = input_01_10 ? (i_input_0 ? 4'sd1 : -4'sd1) : 4'sd0;
    wire signed [3:0] cnt_sum   = $signed({2'b00, cnt}) + cnt_step;
    wire [1:0] cnt_next =
        (cnt_sum < 0)               ? 2'd0 :
        (cnt_sum > $signed({2'b00, CNT_MAX})) ? CNT_MAX :
                                      cnt_sum[1:0];

    // ---- max_rc logic ----
    wire d_enable  = sync_0 ^ sync_1;
    wire and_gate  = sync_1 & d_enable;
    wire dff_next  = d_enable ? and_gate : dff;

    assign o_max = dff ? i_input_1 : i_input_0;   // OLD dff
    assign o_arg = dff_next;                // NEW dff (post-update)

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
