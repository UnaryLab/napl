`timescale 1ns/1ps
`default_nettype none
// Unipolar sqrt_tracejkff equivalent: output=trace|input and
// trace_next=(~trace)&output. Output is combinational (pp_delay=0).
// Active-low reset clears trace.
module sqrt_tracejkff_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (trace=0)
    input  wire i_input,     // input spike stream
    output wire o_out     // square-root spike stream
);
    reg trace;

    // Combinational output: trace OR input. trace is the JK-FF q from cycle t-1.
    assign o_out = trace | i_input;

    // JK-FF trace update with J=o_out, K=1: Q' = (~Q & J) = (~trace) & o_out.
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            trace <= 1'b0;
        else
            trace <= (~trace) & o_out;
    end
endmodule
`default_nettype wire
