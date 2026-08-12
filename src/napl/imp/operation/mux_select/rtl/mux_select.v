`timescale 1ns/1ps
`default_nettype none

// 2-to-1 stream multiplexer: combinational select of one of two spikes.
// The select bit is polarity-agnostic, so one bare module serves both polarities.
// Verify from src/napl/imp with: make test OP=mux_select

module mux_select (
    input  wire i_select,
    input  wire i_input_a,
    input  wire i_input_b,
    output wire o_output
);
    assign o_output = i_select ? i_input_a : i_input_b;
endmodule

`default_nettype wire
