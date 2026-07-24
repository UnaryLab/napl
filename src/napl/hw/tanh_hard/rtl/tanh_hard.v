`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// tanh_hard -- streaming hard tanh, an identity pass-through (combinational).
//
// RTL counterpart of napl.sim.operation.tanh_hard (src/napl/sim/operation/tanh_hard.py).
// The Python forward() is `return input`: no datapath state (the timestep
// counter advances in napl_base.__call__), the input spike passes unchanged.
// It works for both unipolar and bipolar spike streams, so there is no polarity
// split and the module is the bare op name.
//
//   o_out = i_in        (identity)
//
// Stateless and combinational: the output is the input in the same cycle, so
// pp_delay = 0.
//==============================================================================
module tanh_hard (
    input  wire i_in,    // input spike stream
    output wire o_out    // output spike (identity)
);
    assign o_out = i_in;
endmodule
`default_nettype wire
