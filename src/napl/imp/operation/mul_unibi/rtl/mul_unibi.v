`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_unibi -- unipolar x bipolar multiplier with a divider flip-flop.
//
// RTL counterpart of napl.sim.operation.mul_unibi (src/napl/sim/operation/mul_unibi.py).
// i_input_u carries the unipolar magnitude, i_input_b the bipolar signed value,
// and o_output is bipolar: v_z = p_u * (2 * p_b - 1). The port polarities are
// fixed by the operation, so there is no polarity variant.
//
// Two disjoint legs give p_z = p_u * p_b + (1 - p_u) / 2:
//
//   div_event = ~i_input_u                       // one divider event per u zero
//   o_output  = (i_input_u & i_input_b) | (div_event & q)
//   q'        = q ^ div_event                    // fires on every second event
//
// The output is combinational over the current inputs and the held state, so
// pp_delay is 0; each posedge i_clk advances one forward() timestep.
// i_rst_n (active-low) maps to the Python reset(): q <- 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=mul_unibi
//==============================================================================


module mul_unibi (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_u,   // unipolar spike stream
    input  wire i_input_b,   // bipolar spike stream
    output wire o_output     // bipolar product spike
);
    reg  q;
    wire div_event;

    assign div_event = ~i_input_u;
    assign o_output = (i_input_u & i_input_b) | (div_event & q);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            q <= 1'b0;
        else
            q <= q ^ div_event;
    end
endmodule
`default_nettype wire
