`timescale 1ns / 1ps
`default_nettype none

// min_tc -- temporal-coded min via a 2-input AND gate.
// Temporal-coded streams are 1s followed by 0s, so the bitwise AND of two
// streams is the elementwise min. Purely combinational (pp_delay = 0).
// Verify from src/napl/imp with: make test OP=min_tc

module min_tc (
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);

    assign o_output = i_input_0 & i_input_1;

endmodule

`default_nettype wire
