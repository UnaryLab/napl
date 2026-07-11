`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_and_bipolar -- unary/stochastic-computing multiply, bipolar (combinational).
//
// RTL counterpart of napl.operation.mul_and with config polarity='bipolar'
// (src/napl/operation/mul.py).  One spike from each stream per cycle; no state.
//
//   o_out = ~(i_in_0 ^ i_in_1)     (XNOR)
//
// Reference: uGEMM: Unary Computing (Architecture) for GEMM Applications.
//==============================================================================
module mul_and_bipolar (
    input  wire i_in_0,   // spike stream 0
    input  wire i_in_1,   // spike stream 1
    output wire o_out     // product spike
);
    assign o_out = ~(i_in_0 ^ i_in_1);
endmodule
`default_nettype wire
