`timescale 1ns/1ps
`default_nettype none
// Unipolar sqrt_tracejkff equivalent: output=trace|input and
// trace_next=(~trace)&shuffled, where shuffled is the output stream reordered by
// a decorr shuffle buffer. Output is combinational (pp_delay=0).
// Active-low reset clears trace and the shuffle buffer.


module sqrt_tracejkff_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (trace=0)
    input  wire i_input,     // input spike stream
    output wire o_output  // square-root spike stream
);
    reg  trace;
    wire shuffled;

    // Combinational output: trace OR input. trace is the JK-FF q from cycle t-1.
    assign o_output = trace | i_input;

    // The JK-FF holds p_trace = u / (u + 1) only while its J stream is
    // independent of its own stored state; the shuffle buffer reorders the
    // output stream without changing its rate to cut that path. Its sizing is
    // the depth-4, period-256 configuration the Python op fixes in __init__, so
    // the instance takes the module's own defaults and no parameter is passed
    // down. The second stream is fed the same bit and its output is unused; the
    // Python op reads only the first shuffled stream.
    decorr u_decorr (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_input_0  (o_output),
        .i_input_1  (o_output),
        .o_output_0 (shuffled),
        .o_output_1 ()
    );

    // JK-FF trace update with J=shuffled, K=1: Q' = (~Q & J).
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            trace <= 1'b0;
        else
            trace <= (~trace) & shuffled;
    end
endmodule
`default_nettype wire
