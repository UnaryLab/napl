`timescale 1ns/1ps
`default_nettype none

// FSM exp(-2*GAIN*x): bipolar spike input, unipolar spike output.
// State uses an asynchronous active-low reset to the Python reset value.
// Verify from src/napl/imp with: make test OP=exp_n2g

module exp_n2g #(
    parameter integer DEPTH = 5, // inherited from config['depth']; tb overrides via `GEN_DEPTH
    parameter integer GAIN = 1   // inherited from config['gain']; tb overrides via `GEN_GAIN
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    localparam [DEPTH-1:0] CNT_MAX = {DEPTH{1'b1}};
    localparam [DEPTH-1:0] CNT_INIT = {
        1'b1, {(DEPTH-1){1'b0}}
    };
    localparam [DEPTH:0] FULL_SCALE = {
        1'b1, {DEPTH{1'b0}}
    };
    localparam [DEPTH:0] THRESHOLD_EXT =
        FULL_SCALE - GAIN[DEPTH:0];
    localparam [DEPTH-1:0] THRESHOLD =
        THRESHOLD_EXT[DEPTH-1:0];

    reg [DEPTH-1:0] cnt;

    assign o_out = (cnt < THRESHOLD);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cnt <= CNT_INIT;
        else if (i_input && cnt < CNT_MAX)
            cnt <= cnt + 1'b1;
        else if (!i_input && cnt > 0)
            cnt <= cnt - 1'b1;
    end
endmodule

`default_nettype wire
