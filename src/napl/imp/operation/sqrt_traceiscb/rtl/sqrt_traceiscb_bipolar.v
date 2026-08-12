`timescale 1ns/1ps
`default_nettype none
// Bipolar sqrt_traceiscb equivalent with depth-2 cordiv using Sobol indices
// [0,1] and a bi2uni accumulator of width 3, clamped to [-4, 3] as the model
// builds it, and a decorr shuffle buffer on the cordiv dividend path.
// Output is combinational (pp_delay=0).
// Active-low reset clears trace, dff, accumulator, index, and the shuffle
// buffer, and loads the cordiv buffer with the model's alternating [0, 1] rows.


module sqrt_traceiscb_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_output  // output spike stream (combinational, 0-cycle latency)
);
    // Registered state. acc is signed over the width-3 bi2uni clamp [-4, 3];
    // 4 signed bits also hold the reachable pre-clamp range [-5, 1].
    reg trace_q;          // self.trace
    reg dff_q;            // _unipolar_trace dff
    reg signed [3:0] acc_q;  // bi2uni accumulator
    reg buf0_q;           // cordiv buffer_q[0]
    reg buf1_q;           // cordiv buffer_q[1]
    reg idx_q;            // cordiv idx (rand_seq [0,1] -> buf0/buf1)

    wire output_bit = trace_q | i_input;

    // bi2uni: acc + (2*output - 1), clamped to [-4, 3]. Only the lower arm is
    // ever entered: the reachable pre-clamp range is [-5, 1], whose upper end
    // sits below 4'sd3. The upper arm is dead logic kept because it mirrors the
    // two-sided clamp_ the Python model applies.
    wire signed [3:0] acc_step = output_bit ? (acc_q + 4'sd1) : (acc_q - 4'sd1);
    wire signed [3:0] acc_clamped =
        (acc_step > 4'sd3)  ? 4'sd3  :
        (acc_step < -4'sd4) ? -4'sd4 : acc_step;
    wire out_bit = (acc_clamped >= 4'sd1) ? 1'b1 : 1'b0;
    // The emitted spike is subtracted, so the stored value stays in [-4, 0].
    wire signed [3:0] acc_next = acc_clamped - {3'b000, out_bit};

    wire shuffled;

    // The cordiv trace path holds p_trace = u / (u + 1) only while u is
    // independent of the alternating dividend gate; the bi2uni stream is
    // periodic and phase-locks to that period-2 gate, so the divider emits a
    // fixed 0101 and the trace pins at 0.5 regardless of the input. The shuffle
    // buffer reorders u without changing its rate to break the phase lock. Its
    // sizing is the depth-4, period-256 configuration the Python op fixes in
    // __init__, so the instance takes the module's own defaults and no parameter
    // is passed down. The second stream is fed the same bit and its output is
    // unused; the Python op reads only the first shuffled stream.
    decorr u_decorr (
        .i_clk      (i_clk),
        .i_rst_n    (i_rst_n),
        .i_input_0  (out_bit),
        .i_input_1  (out_bit),
        .o_output_0 (shuffled),
        .o_output_1 ()
    );

    wire dff_inv  = ~dff_q;
    wire dividend = dff_inv & shuffled;
    wire divisor  = dff_q | dividend;
    wire rand_q   = (idx_q == 1'b0) ? buf0_q : buf1_q;
    wire quotient = divisor ? dividend : rand_q;

    assign o_output = output_bit;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            trace_q <= 1'b0;
            dff_q   <= 1'b0;
            acc_q   <= 4'sd0;
            buf0_q  <= 1'b0;
            buf1_q  <= 1'b1;
            idx_q   <= 1'b0;
        end else begin
            trace_q <= quotient;
            dff_q   <= dff_inv;
            acc_q   <= acc_next;
            idx_q   <= ~idx_q;
            if (divisor) begin
                buf0_q <= buf1_q;
                buf1_q <= quotient;
            end
        end
    end
endmodule
`default_nettype wire
