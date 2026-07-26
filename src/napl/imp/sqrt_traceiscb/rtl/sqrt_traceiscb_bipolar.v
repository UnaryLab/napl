`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sqrt_traceiscb_bipolar -- bipolar bit-serial square root via stochastic bit
// inserting using the in-stream correlation-based division (iscb) cordiv kernel.
//
// RTL counterpart of napl.sim.operation.sqrt_traceiscb (bipolar branch), see
// src/napl/sim/operation/sqrt_traceiscb.py. One input spike per cycle, one output spike per
// cycle. The output is combinational from the current input and the registered
// trace bit, so the input->output latency is 0 (pp_delay = 0).
//
// Per timestep (Python forward(), bipolar):
//   output = ((1 - trace) & input) + trace          == trace | input
//   out    = bi2uni(output)                          (width-2 accumulator)
//   dff_inv  = ~dff
//   dividend = dff_inv & out
//   divisor  = dff | dividend
//   trace'   = cordiv(dividend, divisor)             (depth-2, rand_seq = [0,1])
//   dff'     = dff_inv
//
// bi2uni (width 2): accumulator acc in [-2, 1] (acc_min = -2, acc_max = 1):
//   acc_c = clamp(acc + 2*output - 1, -2, 1)         // 2*output-1 is +1/-1
//   out   = (acc_c >= 1)
//   acc'  = acc_c - out                              // stays in range
//
// cordiv kernel (div_cordiv, depth=2, rand_seq deterministically [0,1]):
//   rand_q   = (idx == 0) ? buf0 : buf1
//   idx'     = ~idx
//   quotient = divisor ? dividend : rand_q
//   if (divisor) { buf1' = buf0; buf0' = quotient; }
//   trace'   = quotient
//
// Reset (active-low i_rst_n) reproduces the model reset() state EXACTLY:
//   trace = 0, dff = 0, acc = 0, buf0 = 0, buf1 = 0, idx = 0.
//==============================================================================
module sqrt_traceiscb_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_out     // output spike stream (combinational, 0-cycle latency)
);
    // Registered state (post-reset() values are all zero). acc is signed,
    // range [-2, 1]; 3 signed bits hold the pre-clamp value [-3, 2].
    reg trace_q;          // self.trace
    reg dff_q;            // unipolar_trace dff
    reg signed [2:0] acc_q;  // bi2uni accumulator
    reg buf0_q;           // cordiv buffer_q[0]
    reg buf1_q;           // cordiv buffer_q[1]
    reg idx_q;            // cordiv idx (rand_seq [0,1] -> buf0/buf1)

    // Combinational datapath for this timestep.
    wire output_bit = trace_q | i_input;            // ((1-trace)&in)+trace

    // bi2uni: acc + (2*output - 1), clamped to [-2, 1].
    wire signed [2:0] acc_step = output_bit ? (acc_q + 3'sd1) : (acc_q - 3'sd1);
    wire signed [2:0] acc_clamped =
        (acc_step > 3'sd1)  ? 3'sd1  :
        (acc_step < -3'sd2) ? -3'sd2 : acc_step;
    wire out_bit = (acc_clamped >= 3'sd1) ? 1'b1 : 1'b0;
    // acc' = acc_clamped - out_bit (provably stays within [-2, 1]).
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
