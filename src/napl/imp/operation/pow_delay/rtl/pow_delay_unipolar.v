`timescale 1ns/1ps
`default_nettype none
// Unipolar delayed power: AND the current input with its N-1 preceding bits.


module pow_delay_unipolar #(
    parameter integer N     = 2,
    parameter integer DEPTH = 1
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_output
);
    localparam integer DELAY_BITS = (N-1) * DEPTH;

    reg [DELAY_BITS-1:0] delay_line;
    wire [N-1:0] chain;

    assign chain[0] = i_input;

    genvar index;
    generate
        for (index = 1; index < N; index = index + 1) begin : g_product
            assign chain[index] = chain[index-1] & delay_line[index*DEPTH-1];
        end
    endgenerate

    generate
        if (DELAY_BITS == 1) begin : g_delay_single
            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    delay_line[0] <= 1'b0;
                else
                    delay_line[0] <= i_input;
            end
        end else begin : g_delay_chain
            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    delay_line <= {DELAY_BITS{1'b0}};
                else
                    delay_line <= {delay_line[DELAY_BITS-2:0], i_input};
            end
        end
    endgenerate

    assign o_output = chain[N-1];
endmodule
`default_nettype wire
