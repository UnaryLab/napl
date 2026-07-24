`timescale 1ns/1ps
`default_nettype none

// FSM tanh: bipolar spike input and output from a saturating counter.
// State uses an asynchronous active-low reset to the Python reset value.
// Verify from src/napl/hw with: make test OP=tanh_pn
module tanh_pn #(
    parameter integer DEPTH = 3
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,
    output wire o_out
);
    localparam [DEPTH-1:0] CNT_MAX = {DEPTH{1'b1}};
    localparam [DEPTH-1:0] CNT_HALF = {
        1'b1, {(DEPTH-1){1'b0}}
    };

    reg [DEPTH-1:0] cnt;

    assign o_out = (cnt >= CNT_HALF);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cnt <= CNT_HALF;
        else if (i_in && cnt < CNT_MAX)
            cnt <= cnt + 1'b1;
        else if (!i_in && cnt > 0)
            cnt <= cnt - 1'b1;
    end
endmodule

`default_nettype wire
