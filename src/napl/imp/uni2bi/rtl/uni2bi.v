`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// uni2bi -- unipolar-to-bipolar stream converter (clocked, stateful).
//
// RTL counterpart of napl.sim.operation.uni2bi (src/napl/sim/operation/uni2bi.py).
// One input spike per cycle; a signed accumulator turns the unipolar stream
// into a scaled-addition bipolar stream. No polarity variants (bare op name).
//
// Per cycle (matching the Python forward(), a Mealy machine):
//   addend       = i_input ? 2 : 1            // (input + 1)
//   acc_clamped  = clamp(acc + addend, ACC_MIN, ACC_MAX)
//   o_out        = (acc_clamped >= 2)      // combinational in i_input and acc
//   acc <= acc_clamped - (o_out ? 2 : 0)   // state advances each clock
//
// WIDTH is a Verilog parameter inherited from the Python model's config['width']:
// the testbench overrides it with `GEN_WIDTH (emitted by gen/gen_uni2bi.py from
// the same config test_uni2bi.py uses), so the verified hardware always tracks
// the simulator. The default here is only a standalone-elaboration fallback.
//
// ACC_MAX = 2**(WIDTH-1)-1, ACC_MIN = -2**(WIDTH-1) (matching the Python model).
// acc holds [ACC_MIN, ACC_MAX]; the pre-clamp sum acc+addend (addend in {1,2})
// spans [ACC_MIN+1, ACC_MAX+2]. A (WIDTH+1)-bit signed bus (range
// [-2**WIDTH, 2**WIDTH-1]) covers both, so every signed bus is [WIDTH:0].
// o_out is combinational, so it tracks the same-cycle input (pp_delay = 0); the
// only register is acc, reset to 0 to match reset().
//
// Reference: "In-Stream Correlation-Based Division and Bit-Inserting Square
// Root in Stochastic Computing".
//==============================================================================
module uni2bi #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,   // active-low reset -> Python reset(): acc = 0
    input  wire i_input,      // input spike (unipolar stream)
    output wire o_out      // output spike (bipolar stream)
);
    // acc range [ACC_MIN, ACC_MAX]; a (WIDTH+1)-bit signed bus also covers the
    // pre-clamp sum acc+addend (addend in {1,2}).
    localparam signed [WIDTH:0] ACC_MAX =  (1 << (WIDTH-1)) - 1;
    localparam signed [WIDTH:0] ACC_MIN = -(1 << (WIDTH-1));

    reg signed [WIDTH:0] acc;

    // addend = (input + 1): 1 when i_input=0, 2 when i_input=1. The unsized signed
    // decimal constants are sign-extended into the [WIDTH:0] signed context.
    wire signed [WIDTH:0] addend  = i_input ? 2 : 1;
    wire signed [WIDTH:0] sum     = acc + addend;
    // clamp(sum, ACC_MIN, ACC_MAX)
    wire signed [WIDTH:0] clamped = (sum > ACC_MAX) ? ACC_MAX :
                                    (sum < ACC_MIN) ? ACC_MIN : sum;

    assign o_out = (clamped >= 2);

    // acc_next = clamped - 2*out
    wire signed [WIDTH:0] acc_nxt = o_out ? (clamped - 2) : clamped;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {(WIDTH+1){1'b0}};
        else
            acc <= acc_nxt;
    end
endmodule
`default_nettype wire
