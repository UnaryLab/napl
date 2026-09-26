`timescale 1ns/1ps
`default_nettype none
// Ascending threshold sequence backing the `tc` and `temporal` generators of
// napl.sim.operation.encode.gen_num_seq, whose num_seq[i] is i/2**WIDTH: a
// WIDTH-bit up-counter walking 0, 1, ..., 2**WIDTH-1 and wrapping, so o_rand is
// that value scaled by 2**WIDTH.
// Reset lands on 0, the model's num_seq[0], so pp_delay stays 0.
// WIDTH is at least 1: a zero-bit sequence has no points.


module temporalseq #(
    parameter integer WIDTH = 8  // sequence period is 2**WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_en,
    output reg  [WIDTH-1:0] o_rand
);
    generate
        if (WIDTH < 1) begin : g_bad_width
            // An unresolvable module reference makes iverilog fail the build when
            // the sequence would carry no bits.
            ERROR_temporalseq_WIDTH_must_be_at_least_1 u_bad ();
        end
    endgenerate

    localparam [WIDTH-1:0] ONE = 1;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_rand <= {WIDTH{1'b0}};
        end else if (i_en) begin
            o_rand <= o_rand + ONE;
        end
    end
endmodule
`default_nettype wire
