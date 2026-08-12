`timescale 1ns/1ps
`default_nettype none

// Unipolar AND-as-minimum kernel: combinational AND of two spike inputs.
// On maximally-correlated (same number-sequence) unipolar streams this AND
// realizes the elementwise minimum. Verify from src/napl/imp with:
//   make test OP=and_corr
// and_corr is unipolar only, so there is no polarity postfix.

module and_corr (
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);
    assign o_output = i_input_0 & i_input_1;
endmodule

`default_nettype wire
