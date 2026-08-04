`timescale 1ns/1ps
`default_nettype none
// Stream encoder equivalent to napl.sim.operation.encode; pp_delay=0.
// A wrapping counter addresses the Python-generated number-sequence ROM and the
// encoded probability is compared against it with a strict greater-than, matching
// the model's s_t = 1{p > q[(t-1) % L]}.
// Polarity only selects how p is derived from the input value (p = x unipolar,
// p = (x+1)/2 bipolar), so one bare module covers both.
// Generated WIDTH mirrors Python self.width = ceil(log2(config['timestep']));
// FRAC is the shared fixed-point fraction of i_input and the ROM.


module encode #(
    parameter integer WIDTH = 8,  // sequence length is 2**WIDTH; tb overrides via `GEN_WIDTH
    parameter integer FRAC  = 9   // fractional bits of i_input and the ROM; tb overrides via `GEN_FRAC
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire [FRAC:0]    i_input,   // encoded probability p in [0, 2**FRAC]
    output wire             o_spike
);
    // Sequence ROM generated from the model, value = num_seq[i] * 2**FRAC. The vvp
    // cwd is imp/operation/encode/, so the path is relative to that directory.
    reg [FRAC-1:0] num_seq_rom [0:(1<<WIDTH)-1];
    initial $readmemb("vec/encode_rom.hex", num_seq_rom);

    reg  [WIDTH-1:0] seq_idx;
    wire [FRAC-1:0]  num_seq;

    assign num_seq = num_seq_rom[seq_idx];
    assign o_spike = (i_input > {1'b0, num_seq}) ? 1'b1 : 1'b0;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            seq_idx <= {WIDTH{1'b0}};
        else
            seq_idx <= seq_idx + {{(WIDTH-1){1'b0}}, 1'b1};
    end
endmodule
`default_nettype wire
