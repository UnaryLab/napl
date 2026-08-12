`timescale 1ns/1ps
`default_nettype none
// Unipolar sqrt_traceiscb equivalent with a depth-2 cordiv using Sobol indices
// [0,1] and a decorr shuffle buffer on the cordiv dividend path.
// Output is combinational (pp_delay=0).
// Active-low reset clears trace, dff, index, and the shuffle buffer, and loads
// the cordiv buffer with the model's alternating [0, 1] rows.


module sqrt_traceiscb_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_output  // output spike stream (combinational, 0-cycle latency)
);
    // Registered state.
    reg trace_q;   // self.trace
    reg dff_q;     // _unipolar_trace dff
    reg buf0_q;    // cordiv buffer_q[0]
    reg buf1_q;    // cordiv buffer_q[1]
    reg idx_q;     // cordiv idx (selects rand_seq entry: [0,1] -> buf0/buf1)

    wire output_bit = trace_q | i_input;
    wire out_bit    = output_bit;
    wire shuffled;

    // The cordiv trace path holds p_trace = u / (u + 1) only while u is
    // independent of the alternating dividend gate; a periodic u phase-locks to
    // that period-2 gate and the divider emits a fixed 0101. The shuffle buffer
    // reorders u without changing its rate to break the phase lock. Its sizing
    // is the depth-4, period-256 configuration the Python op fixes in __init__,
    // so the instance takes the module's own defaults and no parameter is passed
    // down. The second stream is fed the same bit and its output is unused; the
    // Python op reads only the first shuffled stream.
    decorr u_decorr (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_input_0  (out_bit),
        .i_input_1  (out_bit),
        .o_output_0 (shuffled),
        .o_output_1 ()
    );

    wire dff_inv    = ~dff_q;
    wire dividend   = dff_inv & shuffled;
    wire divisor    = dff_q | dividend;
    wire rand_q     = (idx_q == 1'b0) ? buf0_q : buf1_q;
    wire quotient   = divisor ? dividend : rand_q;

    assign o_output = output_bit;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            trace_q <= 1'b0;
            dff_q   <= 1'b0;
            buf0_q  <= 1'b0;
            buf1_q  <= 1'b1;
            idx_q   <= 1'b0;
        end else begin
            trace_q <= quotient;
            dff_q   <= dff_inv;
            idx_q   <= ~idx_q;
            if (divisor) begin
                buf0_q <= buf1_q;
                buf1_q <= quotient;
            end
        end
    end
endmodule
`default_nettype wire
