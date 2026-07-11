`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sign_abs -- per-timestep sign/magnitude of a bipolar rate-coded spike stream.
//
// RTL counterpart of napl.operation.sign_abs (src/napl/operation/sign_abs.py).
// A saturating up/down accumulator tracks the running balance of the stream:
// each cycle it moves +1 for an input spike of 1 and -1 for 0, clamped to
// [0, ACC_MAX]. The sign/abs OUTPUTS use the UPDATED accumulator and the CURRENT
// input, so they are combinational within the same timestep (pp_delay = 0); the
// register only carries the accumulator forward to the next cycle.
//
//   acc_next = clamp(acc + (i_in ? +1 : -1), 0, ACC_MAX)
//   o_sign   = (acc_next < ACC_MED)        // 1 == negative, 0 == positive
//   o_abs    = o_sign ^ i_in
//
// Reset (active-low i_rst_n) reproduces the Python reset() state EXACTLY:
// acc = ACC_MED = 2**(WIDTH-1). sign_abs has no polarity variants
// (polarity_required=False), so the module is the bare op name.
//
// WIDTH is a Verilog parameter inherited from the Python model's config['width']:
// the testbench overrides it with `GEN_WIDTH (emitted by gen/gen_sign_abs.py from
// the same config test_sign_abs.py uses), so the verified hardware always tracks
// the simulator. ACC_MAX/ACC_MED and the accumulator bus width all derive from
// WIDTH. The default here is only a standalone-elaboration fallback.
//==============================================================================
module sign_abs #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_in,     // bipolar rate-coded input spike
    output wire o_sign,   // 1 == negative, 0 == positive (this timestep)
    output wire o_abs     // magnitude spike
);
    localparam integer ACC_MAX = (1 << WIDTH) - 1;       // saturating maximum
    localparam integer ACC_MED = (1 << (WIDTH - 1));     // midpoint / reset value

    // acc holds the value carried into this timestep; acc_next folds in i_in.
    reg  [WIDTH-1:0] acc;
    reg  [WIDTH-1:0] acc_next;

    always @(*) begin
        if (i_in) begin
            // +1, saturating at ACC_MAX.
            acc_next = (acc == ACC_MAX[WIDTH-1:0]) ? ACC_MAX[WIDTH-1:0]
                                                   : acc + 1'b1;
        end else begin
            // -1, saturating at 0.
            acc_next = (acc == {WIDTH{1'b0}}) ? {WIDTH{1'b0}}
                                              : acc - 1'b1;
        end
    end

    assign o_sign = (acc_next < ACC_MED[WIDTH-1:0]) ? 1'b1 : 1'b0;
    assign o_abs  = o_sign ^ i_in;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= ACC_MED[WIDTH-1:0];
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
