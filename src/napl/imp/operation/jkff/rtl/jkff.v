`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// jkff -- JK flip-flop (clocked, stateful).
//
// RTL counterpart of napl.sim.operation.jkff (src/napl/sim/operation/jkff.py). No
// polarity variants. One spike from each input stream per cycle.
//
//   characteristic eq:  Q' = (J & ~Q) | (~K & Q)
//   the two terms are mutually exclusive, so for Q in {0,1}:  Q' = Q ? ~K : J
//
// i_rst_n (active-low) maps to the Python reset(): q <- 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=jkff
//==============================================================================


module jkff (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_j,   // spike stream J
    input  wire i_input_k,   // spike stream K
    output wire o_q          // flip-flop state spike
);
    reg q;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            q <= 1'b0;
        else
            q <= q ? ~i_input_k : i_input_j;
    end

    assign o_q = q;
endmodule
`default_nettype wire
