`timescale 1ns / 1ps
// min_tc -- temporal-coded min via a 2-input AND gate.
// Temporal-coded streams are 1s followed by 0s, so the bitwise AND of two
// streams is the elementwise min. Purely combinational (pp_delay = 0).
module min_tc (
    input  wire i_in_0,
    input  wire i_in_1,
    output wire o_out
);

    assign o_out = i_in_0 & i_in_1;

endmodule
