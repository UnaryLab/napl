`timescale 1ns/1ps
`default_nettype none
// Bipolar sqrt_tracejkff equivalent with JK trace, width-2 bi2uni, and a decorr
// shuffle buffer between the bi2uni output and the JK input.
// Output is combinational (pp_delay=0); trace and acc update each posedge.
// Active-low reset clears trace, acc, and the shuffle buffer.


module sqrt_tracejkff_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset() (trace=0, acc=0)
    input  wire i_input,     // input spike stream
    output wire o_output  // square-root spike stream
);
    reg               trace;
    reg  signed [2:0] acc;     // bi2uni accumulator, range [-2, 1]

    // trace is the JK-FF output from cycle t-1.
    assign o_output = trace | i_input;

    // acc_sum = acc + 2*output - 1, then clamp to [-2, 1].
    wire signed [3:0] acc_sum = $signed({acc[2], acc}) + (o_output ? 4'sd1 : -4'sd1);
    wire signed [3:0] acc_clamped =
        (acc_sum > 4'sd1)  ? 4'sd1  :
        (acc_sum < -4'sd2) ? -4'sd2 :
                             acc_sum;
    wire out_uni = (acc_clamped >= 4'sd1);
    // acc' remains in [-2, 1] and fits a 3-bit signed value.
    wire signed [2:0] acc_next = acc_clamped[2:0] - (out_uni ? 3'sd1 : 3'sd0);

    // The JK-FF holds p_trace = u / (u + 1) only while its J stream is
    // independent of its own stored state; the shuffle buffer reorders the
    // bi2uni stream without changing its rate to cut that path. Its sizing is
    // the depth-4, period-256 configuration the Python op fixes in __init__, so
    // the instance takes the module's own defaults and no parameter is passed
    // down. The second stream is fed the same bit and its output is unused; the
    // Python op reads only the first shuffled stream.
    wire shuffled;

    decorr u_decorr (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_input_0  (out_uni),
        .i_input_1  (out_uni),
        .o_output_0 (shuffled),
        .o_output_1 ()
    );

    // JK-FF trace update with J=shuffled, K=1: Q' = (~Q & J).
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            trace <= 1'b0;
            acc   <= 3'sd0;
        end else begin
            trace <= (~trace) & shuffled;
            acc   <= acc_next;
        end
    end
endmodule
`default_nettype wire
