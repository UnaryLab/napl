`timescale 1ns/1ps
`default_nettype none

// Unipolar mul_csg equivalent. The WIDTH+1 operand spans [0,2**WIDTH]; an
// input-gated counter addresses the Python-generated number-sequence ROM.
// Output is combinational (pp_delay=0); active-low reset clears the counter.

module mul_csg_unipolar #(
    parameter integer WIDTH = 8   // inherited from ceil(log2(config['timestep'])); tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_input_0,
    input  wire [WIDTH:0]   i_input_1,   // fixed-point operand in [0, 2**WIDTH]
    output wire             o_out
);

    reg  [WIDTH-1:0] seq_idx;
    wire [WIDTH-1:0] num_seq;
    wire             spike;

    // num_seq ROM generated from the model for the inherited WIDTH (one entry per
    // index, value = round(num_seq[i] * 2**WIDTH)); loaded by $readmemb so the
    // table tracks WIDTH. The vvp cwd is
    // imp/operation/mul_csg/, so the path is relative to that dir.
    reg  [WIDTH-1:0] num_seq_rom [0:(1<<WIDTH)-1];
    initial $readmemb("vec/mul_csg_rom.hex", num_seq_rom);

    assign num_seq = num_seq_rom[seq_idx];

    assign spike = (i_input_1 > {1'b0, num_seq}) ? 1'b1 : 1'b0;
    assign o_out = i_input_0 & spike;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            seq_idx <= {WIDTH{1'b0}};
        else
            seq_idx <= seq_idx + {{(WIDTH-1){1'b0}}, i_input_0};
    end

endmodule

`default_nettype wire
