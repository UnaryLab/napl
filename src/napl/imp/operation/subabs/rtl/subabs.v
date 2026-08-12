`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// subabs -- absolute difference of two unipolar streams (combinational).
//
// RTL counterpart of napl.sim.operation.subabs (src/napl/sim/operation/subabs.py).
// The class supports unipolar streams only, so one circuit covers every legal
// polarity and the module carries no polarity postfix. The operands must be
// positively correlated (SCC +1) for the XOR to realize |p_0 - p_1|; that is a
// caller obligation, not something the gate enforces.
//
//   o_output = i_input_0 ^ i_input_1
//
// Stateless, so there is no clock and no reset port; pp_delay is 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=subabs
//==============================================================================


module subabs (
    input  wire i_input_0,   // unipolar spike stream 0
    input  wire i_input_1,   // unipolar spike stream 1
    output wire o_output     // absolute-difference spike
);
    assign o_output = i_input_0 ^ i_input_1;
endmodule
`default_nettype wire
