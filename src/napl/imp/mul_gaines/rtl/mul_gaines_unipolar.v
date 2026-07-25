`timescale 1ns/1ps
`default_nettype none

// Unipolar Gaines multiplier: combinational AND of two spike inputs.
// Verify from src/napl/imp with: make test OP=mul_gaines
module mul_gaines_unipolar (
    input  wire i_in_0,
    input  wire i_in_1,
    output wire o_out
);
    assign o_out = i_in_0 & i_in_1;
endmodule

`default_nettype wire
