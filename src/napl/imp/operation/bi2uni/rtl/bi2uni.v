`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// bi2uni -- bipolar-to-unipolar stream converter (clocked, stateful).
//
// RTL counterpart of napl.sim.operation.bi2uni (src/napl/sim/operation/bi2uni.py).
// One input spike per cycle; a signed accumulator turns the bipolar stream
// into a non-scaled-addition unipolar stream. No polarity variants (bare op
// name).
//
// Per cycle (matching the Python forward(), a Mealy machine):
//   addend       = i_input ? 1 : -1           // (2*input - 1)
//   acc_clamped  = clamp(acc + addend, ACC_MIN, ACC_MAX)
//   o_out        = (acc_clamped >= 1)       // combinational in i_input and acc
//   acc <= acc_clamped - (o_out ? 1 : 0)    // state advances each clock
//
// WIDTH is a Verilog parameter inherited from the Python model's config['width']:
// the testbench overrides it with `GEN_WIDTH (emitted by gen/gen_bi2uni.py from
// the same config test_bi2uni.py uses), so the verified hardware always tracks
// the simulator. The default here is only a standalone-elaboration fallback.
//
// From WIDTH the accumulator bounds are derived exactly as the Python model:
//   ACC_MAX =  2**(WIDTH-1) - 1   (acc.acc_max)
//   ACC_MIN = -2**(WIDTH-1)       (acc.acc_min)
// acc holds [ACC_MIN, ACC_MAX]; the pre-clamp sum spans [ACC_MIN-1, ACC_MAX+1],
// so a (WIDTH+1)-bit signed datapath covers both. o_out is combinational, so it
// tracks the same-cycle input (pp_delay = 0); the only register is acc, reset to
// 0 to match reset().
//
// Reference: "In-Stream Correlation-Based Division and Bit-Inserting Square
// Root in Stochastic Computing".
//==============================================================================
module bi2uni #(
    parameter integer WIDTH = 2   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,   // active-low reset -> Python reset(): acc = 0
    input  wire i_input,      // input spike (bipolar stream)
    output wire o_out      // output spike (unipolar stream)
);
    // (WIDTH+1)-bit signed datapath: holds acc in [ACC_MIN, ACC_MAX] and the
    // pre-clamp sum in [ACC_MIN-1, ACC_MAX+1].
    localparam integer DW = WIDTH + 1;

    // acc bounds derived from WIDTH, matching the Python model's acc_max/acc_min.
    localparam signed [DW-1:0] ACC_MAX =  (1 <<< (WIDTH-1)) - 1;
    localparam signed [DW-1:0] ACC_MIN = -(1 <<< (WIDTH-1));
    localparam signed [DW-1:0] ONE      =  1;

    reg signed [DW-1:0] acc;

    // addend = (2*input - 1): -1 when i_input=0, +1 when i_input=1.
    wire signed [DW-1:0] addend  = i_input ? ONE : -ONE;
    wire signed [DW-1:0] sum     = acc + addend;
    // clamp(sum, ACC_MIN, ACC_MAX)
    wire signed [DW-1:0] clamped = (sum > ACC_MAX) ? ACC_MAX :
                                   (sum < ACC_MIN) ? ACC_MIN : sum;

    assign o_out = (clamped >= ONE);

    // acc_next = clamped - out
    wire signed [DW-1:0] acc_nxt = o_out ? (clamped - ONE) : clamped;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {DW{1'b0}};
        else
            acc <= acc_nxt;
    end
endmodule
`default_nettype wire
