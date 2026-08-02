`timescale 1ns/1ps
`default_nettype none
// Unipolar sqrt_traceiscb equivalent with a depth-2 cordiv using Sobol indices
// [0,1]. Output is combinational (pp_delay=0).
// Active-low reset clears trace, dff, buffer, and index.


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

    wire output_bit = trace_q | i_input;
    wire out_bit    = output_bit;
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
