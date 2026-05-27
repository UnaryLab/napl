`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_and_unipolar -- unary/stochastic-computing multiply, unipolar (combinational).
//
// RTL counterpart of napl.operation.mul_and with config polarity='unipolar'
// (src/napl/operation/mul.py).  One spike from each stream per cycle; no state.
//
//   out = in_0 & in_1        (AND)
//
// Reference: uGEMM: Unary Computing (Architecture) for GEMM Applications.
//==============================================================================
module mul_and_unipolar (
    input  wire in_0,   // spike stream 0
    input  wire in_1,   // spike stream 1
    output wire out     // product spike
);
    assign out = in_0 & in_1;
endmodule
`default_nettype wire
