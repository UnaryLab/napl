`timescale 1ns/1ps
`default_nettype none

// Saturating add of two independent unipolar spike streams: combinational OR.
// Verify from src/napl/imp with: make test OP=or_sat

module or_sat (
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);
    assign o_output = i_input_0 | i_input_1;
endmodule

`default_nettype wire
