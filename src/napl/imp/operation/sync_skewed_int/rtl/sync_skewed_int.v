`timescale 1ns/1ps
`default_nettype none

// Integral skewed synchronizer with a WIDTH-bit saturating spike counter.
// State uses an asynchronous active-low reset to the Python reset value.
// Verify from src/napl/imp with: make test OP=sync_skewed_int

module sync_skewed_int #(
    parameter integer WIDTH = 4  // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_input_1,
    input  wire             i_input_2,
    output wire [WIDTH-1:0] o_out_1,
    output wire             o_out_2
);
    reg [WIDTH-1:0] cnt;

    wire [WIDTH:0] sum_ext = {1'b0, cnt}
        + {{WIDTH{1'b0}}, i_input_1};
    wire [WIDTH-1:0] clipped_sum = sum_ext[WIDTH]
        ? {WIDTH{1'b1}} : sum_ext[WIDTH-1:0];

    assign o_out_1 = i_input_2 ? clipped_sum : {WIDTH{1'b0}};
    assign o_out_2 = i_input_2;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cnt <= {WIDTH{1'b0}};
        else if (i_input_2)
            cnt <= {{(WIDTH-1){1'b0}}, sum_ext[WIDTH]};
        else
            cnt <= clipped_sum;
    end
endmodule

`default_nettype wire
