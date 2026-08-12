`timescale 1ns/1ps
`default_nettype none
// Unipolar x^n: a chain of N-1 Gaines multiplies (AND), each combining the
// running product with a differently delayed copy of the input. Stage k delays
// the input by k cycles (a depth-k dff), so the N factors are mutually
// decorrelated. Output is combinational per timestep (the dff chain holds the
// only state); active-low reset clears every delay line.
// Composes the standalone dff and mul_gaines_unipolar modules (resolved by the -y
// library), matching napl.sim.operation.pow_n's internal wiring.


module pow_n_unipolar #(
    parameter integer N = 2   // inherited from config['n']; tb overrides via `GEN_N
) (
    input  wire i_clk,    // sample clock; one tick == one Python forward()
    input  wire i_rst_n,  // active-low reset -> Python reset() (clears the dff chain)
    input  wire i_input,  // input spike stream
    output wire o_output  // input raised to the N-th power
);
    // chain[0] is the raw input; chain[k] is the product after stage k's multiply;
    // chain[N-1] is the final x^N spike read out this cycle.
    wire [N-1:0] chain;
    assign chain[0] = i_input;

    genvar k;
    generate
        for (k = 1; k < N; k = k + 1) begin : g_stage
            // Stage k's decorrelated factor: the input delayed by k cycles.
            wire delayed;
            dff #(.DEPTH(k)) u_dff (
                .i_clk(i_clk), .i_rst_n(i_rst_n),
                .i_input(i_input), .o_output(delayed)
            );
            mul_gaines_unipolar u_mul (
                .i_input_0(chain[k-1]), .i_input_1(delayed), .o_output(chain[k])
            );
        end
    endgenerate

    assign o_output = chain[N-1];
endmodule
`default_nettype wire
