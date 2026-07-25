`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sync_skewed -- skewed synchronizer of two spike streams (stateful).
//
// RTL counterpart of napl.sim.operation.sync_skewed (src/napl/sim/operation/sync_skewed.py).
// Assumes stream 1's rate <= stream 2's rate. Stream 2 passes through unchanged;
// stream 1 is re-timed toward stream 2 using a saturating WIDTH-bit counter that
// buffers the lead/lag between the two streams.
//
// Per cycle (one Python forward() timestep), with the counter cnt read BEFORE
// it is updated:
//   diff = i_in_1 ^ i_in_2                     // inputs are 01 or 10
//   when diff:
//     i_in_1==1 (push): o_out_1 = (cnt==CNT_MAX); cnt saturates up   (cnt+1)
//     i_in_1==0 (pop) : o_out_1 = (cnt!=0);       cnt saturates down (cnt-1)
//   when !diff (00/11): o_out_1 = i_in_1; cnt unchanged.
// o_out_2 = i_in_2 always. Output is combinational in (i_in_1,i_in_2,cnt), so
// the input->output latency is 0; the counter update is registered.
//
// Reset (active-low i_rst_n) reproduces the Python reset() state: cnt = 0.
//
// sync_skewed has no polarity variants (polarity_required=False), so the module
// is the bare op name with no _unipolar/_bipolar postfix.
//
// WIDTH is a Verilog parameter inherited from the Python model's config['width']:
// the testbench overrides it with `GEN_WIDTH (emitted by gen/gen_sync_skewed.py
// from the same config test_sync_skewed.py uses), so the verified hardware always
// tracks the simulator. The default here is only a standalone-elaboration
// fallback. The saturating bound CNT_MAX = 2**WIDTH - 1 derives from WIDTH.
//==============================================================================
module sync_skewed #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (cnt = 0)
    input  wire i_in_1,   // input spike stream 1 (lower rate)
    input  wire i_in_2,   // input spike stream 2 (higher rate, passes through)
    output wire o_out_1,  // re-timed stream 1
    output wire o_out_2   // stream 2, unchanged
);
    localparam integer CNT_MAX = (1 << WIDTH) - 1;

    reg [WIDTH-1:0] cnt;

    wire diff        = i_in_1 ^ i_in_2;          // 01 or 10
    wire cnt_not_min = (cnt != 0);
    wire cnt_not_max = (cnt != CNT_MAX[WIDTH-1:0]);

    // o_out_1: on 00/11 pass i_in_1 through; on a push emit only when saturated
    // high, on a pop emit only when not empty.
    assign o_out_1 = diff ? (i_in_1 ? ~cnt_not_max : cnt_not_min) : i_in_1;
    assign o_out_2 = i_in_2;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= {WIDTH{1'b0}};
        end else if (diff) begin
            if (i_in_1) begin
                if (cnt_not_max) cnt <= cnt + 1'b1;   // push, saturating
            end else begin
                if (cnt_not_min) cnt <= cnt - 1'b1;   // pop, saturating
            end
        end
    end
endmodule
`default_nettype wire
