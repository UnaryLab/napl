// mul_csg_unipolar -- unary multiply by conditional spike generation (unipolar).
//
// Bit-serial port of napl mul_csg.forward() (unipolar branch). i_in_1 is the
// fixed-point operand round(in_1 * 2**WIDTH) in [0, 2**WIDTH] (WIDTH+1 bits: prob
// 1.0 maps to 2**WIDTH); i_in_0 is the 1-bit input spike stream. A WIDTH-bit
// counter (seq_idx) walks the num_seq ROM (WIDTH-bit values in [0, 2**WIDTH-1]):
//   spike = (i_in_1 > num_seq[seq_idx]); o_out = i_in_0 & spike;
//   seq_idx advances by i_in_0 each cycle (the enable).
// The output is combinational in the CURRENT seq_idx (pp_delay = 0); seq_idx is
// the only state, registered on posedge i_clk and cleared by i_rst_n (matching
// reset(): seq_idx = 0).
//
// WIDTH (= ceil(log2(timestep))) is a Verilog parameter inherited from the
// Python model's config: the testbench overrides it with `GEN_WIDTH (emitted by
// gen/gen_mul_csg.py from the same config), so the verified hardware tracks the
// simulator. WIDTH sizes the seq-index counter, the ROM value width, and the
// i_in_1 operand bus (WIDTH+1 bits, so prob 1.0 -> 2**WIDTH is representable).
// The num_seq ROM is GENERATED from the model for the inherited WIDTH and loaded
// via $readmemb from vec/mul_csg_rom.hex, so the table follows WIDTH.
module mul_csg_unipolar #(
    parameter integer WIDTH = 8   // inherited from ceil(log2(config['timestep'])); tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_in_0,
    input  wire [WIDTH:0]   i_in_1,   // fixed-point operand in [0, 2**WIDTH]
    output wire             o_out
);

    reg  [WIDTH-1:0] seq_idx;
    reg  [WIDTH-1:0] num_seq;
    wire             spike;

    // num_seq ROM generated from the model for the inherited WIDTH (one entry per
    // index, value = round(num_seq[i] * 2**WIDTH)); loaded by $readmemb so the
    // table tracks WIDTH. The vvp cwd is
    // implementation/mul_csg/, so the path is relative to that dir.
    reg  [WIDTH-1:0] num_seq_rom [0:(1<<WIDTH)-1];
    initial $readmemb("vec/mul_csg_rom.hex", num_seq_rom);

    // combinational ROM read in the current seq_idx
    always @(*) begin
        num_seq = num_seq_rom[seq_idx];
    end

    assign spike = (i_in_1 > {1'b0, num_seq}) ? 1'b1 : 1'b0;
    assign o_out = i_in_0 & spike;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            seq_idx <= {WIDTH{1'b0}};
        else
            seq_idx <= seq_idx + {{(WIDTH-1){1'b0}}, i_in_0};
    end

endmodule
