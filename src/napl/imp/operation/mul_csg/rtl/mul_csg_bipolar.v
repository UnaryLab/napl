`timescale 1ns/1ps
`default_nettype none

// Bipolar mul_csg equivalent. The WIDTH+1 operand spans [0,2**WIDTH]; two
// input-gated counters address the Python-generated number-sequence ROM.
// Output is combinational (pp_delay=0); active-low reset clears both counters.
module mul_csg_bipolar #(
    parameter integer WIDTH = 8   // inherited from ceil(log2(config['timestep'])); tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_input_0,
    input  wire [WIDTH:0]   i_input_1,   // fixed-point operand in [0, 2**WIDTH]
    output wire             o_out
);

    reg  [WIDTH-1:0] seq_idx;
    reg  [WIDTH-1:0] seq_idx_inv;
    wire [WIDTH-1:0] num_seq;
    wire [WIDTH-1:0] num_seq_inv;
    wire             spike;
    wire             spike_inv;
    wire             path;
    wire             path_inv;

    // num_seq ROM generated from the model for the inherited WIDTH (one entry per
    // index, value = round(num_seq[i] * 2**WIDTH)); loaded by $readmemb so the
    // table tracks WIDTH. Both read ports
    // (seq_idx and seq_idx_inv) index the SAME array. The vvp cwd is
    // imp/operation/mul_csg/, so the path is relative to that dir.
    reg  [WIDTH-1:0] num_seq_rom [0:(1<<WIDTH)-1];
    initial $readmemb("vec/mul_csg_rom.hex", num_seq_rom);

    assign num_seq = num_seq_rom[seq_idx];
    assign num_seq_inv = num_seq_rom[seq_idx_inv];

    assign spike     = (i_input_1 > {1'b0, num_seq})     ? 1'b1 : 1'b0;
    assign spike_inv = (i_input_1 > {1'b0, num_seq_inv}) ? 1'b1 : 1'b0;
    assign path      = i_input_0 & spike;
    assign path_inv  = (~i_input_0) & (~spike_inv);
    assign o_out     = path | path_inv;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            seq_idx     <= {WIDTH{1'b0}};
            seq_idx_inv <= {WIDTH{1'b0}};
        end else begin
            seq_idx     <= seq_idx     + {{(WIDTH-1){1'b0}},  i_input_0};
            seq_idx_inv <= seq_idx_inv + {{(WIDTH-1){1'b0}}, ~i_input_0};
        end
    end

endmodule

`default_nettype wire
