`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// negate -- bipolar stream negation by spike inversion (combinational).
//
// RTL counterpart of napl.sim.operation.negate (src/napl/sim/operation/negate.py).
// The class supports bipolar streams only, so one circuit covers every legal
// polarity and the module carries no polarity postfix.
//
//   o_output = ~i_input        (p_y = 1 - p_x, so v_y = -v_x)
//
// Stateless, so there is no clock and no reset port; pp_delay is 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=negate
//==============================================================================


module negate (
    input  wire i_input,    // bipolar spike stream
    output wire o_output    // negated bipolar spike stream
);
    assign o_output = ~i_input;
endmodule
`default_nettype wire
