`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sqrt_traceiscb_unipolar -- unipolar bit-serial square root via stochastic bit
// inserting using the in-stream correlation-based division (iscb) cordiv kernel.
//
// RTL counterpart of napl.sim.operation.sqrt_traceiscb (unipolar branch), see
// src/napl/sim/operation/sqrt_traceiscb.py. One input spike per cycle, one output spike per
// cycle. The output is combinational from the current input and the registered
// trace bit, so the input->output latency is 0 (pp_delay = 0).
//
// Per timestep (Python forward(), unipolar):
//   output = ((1 - trace) & input) + trace          == trace | input
//   out    = output                                  (unipolar: no bi2uni)
//   dff_inv  = ~dff
//   dividend = dff_inv & out
//   divisor  = dff | dividend
//   trace'   = cordiv(dividend, divisor)             (depth-2, rand_seq = [0,1])
//   dff'     = dff_inv
//
// cordiv kernel (div_cordiv, depth=2, rand_seq deterministically [0,1]):
//   rand_q   = (idx == 0) ? buf0 : buf1
//   idx'     = ~idx
//   quotient = divisor ? dividend : rand_q
//   if (divisor) { buf1' = buf0; buf0' = quotient; }
//   trace'   = quotient
//
// Reset (active-low i_rst_n) reproduces the model reset() state EXACTLY:
//   trace = 0, dff = 0, buf0 = 0, buf1 = 0, idx = 0.
//==============================================================================
module sqrt_traceiscb_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_out     // output spike stream (combinational, 0-cycle latency)
);
    // Registered state (post-reset() values are all zero).
    reg trace_q;   // self.trace
    reg dff_q;     // unipolar_trace dff
    reg buf0_q;    // cordiv buffer_q[0]
    reg buf1_q;    // cordiv buffer_q[1]
    reg idx_q;     // cordiv idx (selects rand_seq entry: [0,1] -> buf0/buf1)

    // Combinational datapath for this timestep.
    wire output_bit = trace_q | i_input;            // ((1-trace)&in)+trace
    wire out_bit    = output_bit;                // unipolar: out = output
    wire dff_inv    = ~dff_q;
    wire dividend   = dff_inv & out_bit;
    wire divisor    = dff_q | dividend;
    wire rand_q     = (idx_q == 1'b0) ? buf0_q : buf1_q;
    wire quotient   = divisor ? dividend : rand_q;

    assign o_out = output_bit;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            trace_q <= 1'b0;
            dff_q   <= 1'b0;
            buf0_q  <= 1'b0;
            buf1_q  <= 1'b0;
            idx_q   <= 1'b0;
        end else begin
            trace_q <= quotient;
            dff_q   <= dff_inv;
            idx_q   <= ~idx_q;
            if (divisor) begin
                buf1_q <= buf0_q;
                buf0_q <= quotient;
            end
        end
    end
endmodule
`default_nettype wire
