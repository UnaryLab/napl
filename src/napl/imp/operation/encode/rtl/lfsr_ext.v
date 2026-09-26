`timescale 1ns/1ps
`default_nettype none
// Online extended LFSR matching napl.sim.operation.encode._get_lfsr_ext_seq: the
// maximal cycle of the feedback polynomial with the missing all-zero state
// inserted right after the state whose value is 1, giving all 2**WIDTH distinct
// values over one period. o_rand = int(state, 2), the model's num_seq value
// scaled by 2**WIDTH.
// The insertion is the de Bruijn modification of the feedback: the tap XOR is
// inverted whenever the upper state bits are all zero, which is exactly the
// states 1 and 0, so state 1 shifts to 0 and state 0 shifts to what state 1 fed
// on the unmodified cycle. TAPS is the tap mask recovered from the model
// sequence by the generator and SEED is the model's step-0 state, which reset
// restores so pp_delay stays 0.
// WIDTH is at least 2: the underlying maximal cycle needs a feedback polynomial,
// which is the bound get_lfsr_seq raises on, and the insertion test reads the
// part select o_rand[WIDTH-1:1].


module lfsr_ext #(
    parameter integer WIDTH = 8,        // sequence period is 2**WIDTH
    parameter [WIDTH-1:0] TAPS = 113,   // feedback tap mask, bit i selects o_rand[i]
    parameter [WIDTH-1:0] SEED = 1      // post-reset state, the model's num_seq[0]
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_en,
    output reg  [WIDTH-1:0] o_rand
);
    generate
        if (WIDTH < 2) begin : g_bad_width
            // An unresolvable module reference makes iverilog fail the build when
            // the width carries no feedback polynomial.
            ERROR_lfsr_ext_WIDTH_must_be_at_least_2 u_bad ();
        end
    endgenerate

    wire insert   = ~(|o_rand[WIDTH-1:1]);
    wire feedback = (^(o_rand & TAPS)) ^ insert;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            o_rand <= SEED;
        end else if (i_en) begin
            o_rand <= {feedback, o_rand[WIDTH-1:1]};
        end
    end
endmodule
`default_nettype wire
