`timescale 1ns/1ps
`default_nettype none

// Unipolar mul_ugemm equivalent. The WIDTH+1 operand spans [0,2**WIDTH]; an
// input-gated online Sobol generator supplies the number sequence.
// Output is combinational (pp_delay=0); active-low reset restarts the sequence.

module mul_ugemm_unipolar #(
    parameter integer WIDTH = 8   // inherited from ceil(log2(config['timestep'])); tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_input_0,
    input  wire [WIDTH:0]   i_input_1,   // fixed-point operand in [0, 2**WIDTH]
    output wire             o_output
);

    wire [WIDTH-1:0] num_seq;
    wire             spike;

    // The number sequence is produced online by the Sobol recurrence over a
    // WIDTH-entry direction-vector table, advanced by the input spike so the
    // emitted value is the model's num_seq at the current sequence index. The
    // vvp cwd is imp/operation/mul_ugemm/, so the path is relative to that dir.
    sobol #(
        .WIDTH       (WIDTH),
        .DIRVEC_FILE ("vec/mul_ugemm_dv.hex")
    ) u_seq (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_en    (i_input_0),
        .o_rand  (num_seq)
    );

    assign spike = (i_input_1 > {1'b0, num_seq}) ? 1'b1 : 1'b0;
    assign o_output = i_input_0 & spike;

endmodule

`default_nettype wire
