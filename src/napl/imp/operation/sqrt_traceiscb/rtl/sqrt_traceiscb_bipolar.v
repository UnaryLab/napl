`timescale 1ns/1ps
`default_nettype none
// Bipolar sqrt_traceiscb equivalent with width-2 bi2uni and depth-2 cordiv
// using Sobol indices [0,1]. Output is combinational (pp_delay=0).
// Active-low reset clears trace, dff, accumulator, buffer, and index.


module sqrt_traceiscb_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_out     // output spike stream (combinational, 0-cycle latency)
);
    // Registered state (post-reset() values are all zero). acc is signed,
    // range [-2, 1]; 3 signed bits hold the pre-clamp value [-3, 2].
    reg trace_q;          // self.trace
    reg dff_q;            // _unipolar_trace dff
    reg signed [2:0] acc_q;  // bi2uni accumulator
    reg buf0_q;           // cordiv buffer_q[0]
    reg buf1_q;           // cordiv buffer_q[1]
    reg idx_q;            // cordiv idx (rand_seq [0,1] -> buf0/buf1)

    wire output_bit = trace_q | i_input;

    // bi2uni: acc + (2*output - 1), clamped to [-2, 1].
    wire signed [2:0] acc_step = output_bit ? (acc_q + 3'sd1) : (acc_q - 3'sd1);
    wire signed [2:0] acc_clamped =
        (acc_step > 3'sd1)  ? 3'sd1  :
        (acc_step < -3'sd2) ? -3'sd2 : acc_step;
    wire out_bit = (acc_clamped >= 3'sd1) ? 1'b1 : 1'b0;
    // acc' remains in [-2, 1].
    wire signed [2:0] acc_next = acc_clamped - {2'b00, out_bit};

    wire dff_inv  = ~dff_q;
    wire dividend = dff_inv & out_bit;
    wire divisor  = dff_q | dividend;
    wire rand_q   = (idx_q == 1'b0) ? buf0_q : buf1_q;
    wire quotient = divisor ? dividend : rand_q;

    assign o_out = output_bit;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            trace_q <= 1'b0;
            dff_q   <= 1'b0;
            acc_q   <= 3'sd0;
            buf0_q  <= 1'b0;
            buf1_q  <= 1'b0;
            idx_q   <= 1'b0;
        end else begin
            trace_q <= quotient;
            dff_q   <= dff_inv;
            acc_q   <= acc_next;
            idx_q   <= ~idx_q;
            if (divisor) begin
                buf1_q <= buf0_q;
                buf0_q <= quotient;
            end
        end
    end
endmodule
`default_nettype wire
