`timescale 1ns/1ps
`default_nettype none
// Bipolar square_dff equivalent: XNOR the current input with its DEPTH-delayed
// copy. Output is combinational (pp_delay=0); active-low reset clears the delay.


module square_dff_bipolar #(
    parameter integer DEPTH = 1   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,    // sample clock; one tick == one Python forward()
    input  wire i_rst_n,  // active-low reset -> Python reset() (clears delay reg)
    input  wire i_input,     // input spike stream
    output wire o_output  // squared product spike
);
    // depth-DEPTH delay line: in_d[0] is the oldest cell (the delayed copy used
    // this cycle); in_d[DEPTH-1] holds the most recently written input.
    reg [DEPTH-1:0] in_d;

    genvar index;
    generate
        for (index = 0; index < DEPTH; index = index + 1) begin : g_register
            if (index == DEPTH-1) begin : g_tail
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        in_d[index] <= 1'b0;
                    else
                        in_d[index] <= i_input;
                end
            end else begin : g_body
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        in_d[index] <= 1'b0;
                    else
                        in_d[index] <= in_d[index+1];
                end
            end
        end
    endgenerate

    // Combinational XNOR of the current input and its oldest delayed copy.
    assign o_output = ~(i_input ^ in_d[0]);
endmodule
`default_nettype wire
