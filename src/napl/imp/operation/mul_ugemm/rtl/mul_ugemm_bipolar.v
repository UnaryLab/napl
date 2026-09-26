`timescale 1ns/1ps
`default_nettype none

// Bipolar mul_ugemm equivalent. The WIDTH+1 operand spans [0,2**WIDTH]; two
// online Sobol generators walk the same sequence, one advanced by the input
// spike and one by its complement.
// Output is combinational (pp_delay=0); active-low reset restarts both.

module mul_ugemm_bipolar #(
    parameter integer WIDTH = 8   // inherited from ceil(log2(config['timestep'])); tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_input_0,
    input  wire [WIDTH:0]   i_input_1,   // fixed-point operand in [0, 2**WIDTH]
    output wire             o_output
);

    wire [WIDTH-1:0] num_seq;
    wire [WIDTH-1:0] num_seq_inv;
    wire             spike;
    wire             spike_inv;
    wire             path;
    wire             path_inv;

    // Both paths walk the same number sequence, produced online by the Sobol
    // recurrence over a WIDTH-entry direction-vector table: one generator is
    // advanced by the input spike and one by its complement, so each emits the
    // model's num_seq at its own sequence index. Both read the same table. The
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

    sobol #(
        .WIDTH       (WIDTH),
        .DIRVEC_FILE ("vec/mul_ugemm_dv.hex")
    ) u_seq_inv (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_en    (~i_input_0),
        .o_rand  (num_seq_inv)
    );

    assign spike     = (i_input_1 > {1'b0, num_seq})     ? 1'b1 : 1'b0;
    assign spike_inv = (i_input_1 > {1'b0, num_seq_inv}) ? 1'b1 : 1'b0;
    assign path      = i_input_0 & spike;
    assign path_inv  = (~i_input_0) & (~spike_inv);
    assign o_output     = path | path_inv;

endmodule

`default_nettype wire
