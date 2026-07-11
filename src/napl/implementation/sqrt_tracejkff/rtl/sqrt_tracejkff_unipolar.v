`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sqrt_tracejkff_unipolar -- unipolar bit-inserting square root via JK-FF trace.
//
// RTL counterpart of napl.operation.sqrt_tracejkff (forward(), unipolar branch)
// in src/napl/operation/sqrt.py. Per timestep t (one posedge i_clk):
//
//   output = trace | i_in           // trace is the JK-FF state from cycle t-1
//   trace' = (~trace) & output      // JK update: J=output, K=1 => Q' = ~Q & J
//
// The output is combinational in i_in given the current trace register, so the
// input->output latency is 0 (pp_delay = 0). Only the trace register is clocked.
//
// Reset (active-low i_rst_n) maps to the Python reset(): jkff.q = 0.
//==============================================================================
module sqrt_tracejkff_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (trace=0)
    input  wire i_in,     // input spike stream
    output wire o_out     // square-root spike stream
);
    reg trace;

    // Combinational output: trace OR input. trace is the JK-FF q from cycle t-1.
    assign o_out = trace | i_in;

    // JK-FF trace update with J=o_out, K=1: Q' = (~Q & J) = (~trace) & o_out.
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            trace <= 1'b0;
        else
            trace <= (~trace) & o_out;
    end
endmodule
`default_nettype wire
