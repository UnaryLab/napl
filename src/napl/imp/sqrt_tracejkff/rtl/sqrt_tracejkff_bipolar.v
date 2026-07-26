`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sqrt_tracejkff_bipolar -- bipolar bit-inserting square root via JK-FF trace.
//
// RTL counterpart of napl.sim.operation.sqrt_tracejkff (forward(), bipolar branch)
// in src/napl/sim/operation/sqrt_tracejkff.py. Per timestep t (one posedge i_clk):
//
//   output  = trace | i_input                  // trace = JK-FF state from t-1
//   out_uni = bi2uni(output)                 // width-2 bipolar->unipolar
//   trace'  = (~trace) & out_uni             // JK update: J=out_uni, K=1
//
// bi2uni (width=2, acc in [acc_min=-2, acc_max=1]):
//   acc_n   = clamp(acc + 2*output - 1, -2, 1)
//   out_uni = (acc_n >= 1)
//   acc'    = acc_n - out_uni                // stays in range
//
// The output is combinational in i_input given the current trace register, so the
// input->output latency is 0 (pp_delay = 0). The trace and acc registers are
// clocked and updated each cycle.
//
// Reset (active-low i_rst_n) maps to the Python reset(): jkff.q=0, bi2uni acc=0.
//==============================================================================
module sqrt_tracejkff_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (trace=0, acc=0)
    input  wire i_input,     // input spike stream
    output wire o_out     // square-root spike stream
);
    reg               trace;
    reg  signed [2:0] acc;     // bi2uni accumulator, range [-2, 1]

    // Combinational output: trace OR input. trace is the JK-FF q from cycle t-1.
    assign o_out = trace | i_input;

    // bi2uni step on this cycle's output.
    // acc_sum = acc + 2*output - 1, then clamp to [-2, 1].
    wire signed [3:0] acc_sum = $signed({acc[2], acc}) + (o_out ? 4'sd1 : -4'sd1);
    wire signed [3:0] acc_clamped =
        (acc_sum > 4'sd1)  ? 4'sd1  :
        (acc_sum < -4'sd2) ? -4'sd2 :
                             acc_sum;
    wire out_uni = (acc_clamped >= 4'sd1);
    // acc' = acc_clamped - out_uni (provably stays within [-2, 1], fits 3-bit signed).
    wire signed [2:0] acc_next = acc_clamped[2:0] - (out_uni ? 3'sd1 : 3'sd0);

    // JK-FF trace update with J=out_uni, K=1: Q' = (~Q & J).
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            trace <= 1'b0;
            acc   <= 3'sd0;
        end else begin
            trace <= (~trace) & out_uni;
            acc   <= acc_next;
        end
    end
endmodule
`default_nettype wire
