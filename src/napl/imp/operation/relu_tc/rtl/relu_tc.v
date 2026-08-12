`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// relu_tc -- bipolar temporal-coded ReLU.
//
// relu(x) is the temporal maximum of the input against a stream carrying zero,
// so the module generates that zero reference internally and ORs it with the
// input. A bipolar zero falls at the midpoint of the codeword, so the reference
// bit is the inverted MSB of a WIDTH-bit cycle counter: 1 for the first
// 2**(WIDTH-1) cycles and 0 for the rest, wrapping with the codeword. The output is
// combinational in the arrival cycle and the counter commits on the clock edge,
// so one vector row is one timestep.
//==============================================================================


module relu_tc #(
    parameter integer WIDTH = 8   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_output
);
    // Completed timesteps modulo the codeword length 2**WIDTH.
    reg [WIDTH-1:0] cycle;

    // Current bit of the internal stream encoding zero.
    wire reference;

    assign reference = ~cycle[WIDTH-1];
    assign o_output = i_input | reference;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            cycle <= {WIDTH{1'b0}};
        else
            cycle <= cycle + 1'b1;
    end
endmodule
`default_nettype wire
