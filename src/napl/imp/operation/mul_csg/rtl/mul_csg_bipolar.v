`timescale 1ns/1ps
`default_nettype none

// mul_csg_bipolar -- unary multiply by conditional spike generation (bipolar).
//
// Bit-serial port of napl mul_csg.forward() (bipolar branch). i_input_1 is the
// fixed-point operand round((in_1+1)/2 * 2**WIDTH) in [0, 2**WIDTH] (WIDTH+1 bits:
// prob 1.0 maps to 2**WIDTH); i_input_0 is the 1-bit input spike. Two independent
// ROM-walking counters (WIDTH-bit ROM values in [0, 2**WIDTH-1]) drive the paths:
//   spike     = (i_input_1 > num_seq[seq_idx]);     path     = i_input_0 & spike
//   spike_inv = (i_input_1 > num_seq[seq_idx_inv]); path_inv = ~i_input_0 & ~spike_inv
//   o_out = path | path_inv
//   seq_idx advances by i_input_0, seq_idx_inv by ~i_input_0 (their enables).
// Output is combinational in the CURRENT counters (pp_delay = 0); both counters
// are registered on posedge i_clk and cleared to 0 by i_rst_n (matching reset()).
//
// WIDTH (= ceil(log2(timestep))) is a Verilog parameter inherited from the
// Python model's config: the testbench overrides it with `GEN_WIDTH (emitted by
// gen/gen_mul_csg.py from the config), so the verified hardware tracks the
// simulator. WIDTH sizes both seq-index counters, the ROM value width, and the
// i_input_1 operand bus (WIDTH+1 bits). The num_seq ROM is GENERATED from the model
// for the inherited WIDTH and loaded via $readmemb from vec/mul_csg_rom.hex; the
// bipolar branch reuses the SAME table at both indices, so one ROM file serves
// both read ports.
// Verify from src/napl/imp with: make test OP=mul_csg
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
