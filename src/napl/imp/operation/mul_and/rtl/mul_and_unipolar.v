`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_and_unipolar -- unary/stochastic-computing multiply, unipolar (combinational).
//
// RTL counterpart of napl.sim.operation.mul_and with config polarity='unipolar'
// (src/napl/sim/operation/mul_and.py).  One spike from each stream per cycle; no state.
//
//   o_out = i_input_0 & i_input_1        (AND)
//
// Reference: uGEMM: Unary Computing (Architecture) for GEMM Applications.
//==============================================================================


module mul_and_unipolar (
    input  wire i_input_0,   // spike stream 0
    input  wire i_input_1,   // spike stream 1
    output wire o_out     // product spike
);
    assign o_out = i_input_0 & i_input_1;
endmodule
`default_nettype wire
