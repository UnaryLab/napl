`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// max_tc -- temporal-coded max via OR gate (combinational).
//
// RTL counterpart of napl.sim.operation.max_tc (src/napl/sim/operation/max_tc.py).
// Temporal-coded streams start with 1s followed by 0s; the max of two such
// streams is their bitwise OR. One spike from each stream per cycle; no state,
// no polarity variant.
//
//   o_out = i_input_0 | i_input_1        (OR)
//==============================================================================


module max_tc (
    input  wire i_input_0,   // spike stream 0
    input  wire i_input_1,   // spike stream 1
    output wire o_out     // max spike
);
    assign o_out = i_input_0 | i_input_1;
endmodule
`default_nettype wire
