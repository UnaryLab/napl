`timescale 1ns/1ps
`default_nettype none
// Online Fibonacci LFSR matching napl.sim.operation.encode.get_lfsr_seq: the
// state shifts right by one bit each cycle and the vacated top bit takes the XOR
// of the tapped state bits, so o_rand walks the maximal cycle of the feedback
// polynomial with o_rand = int(state, 2), the model's num_seq value scaled by
// 2**WIDTH.
// TAPS is the tap mask recovered from the model sequence by the generator, bit i
// selecting o_rand[i]; SEED is the model's step-0 state, which reset restores so
// pp_delay stays 0.
// The model emits 2**WIDTH samples of a cycle whose period is 2**WIDTH - 1, so
// its opening state appears at both index 0 and index 2**WIDTH - 1. CNT counts
// the emitted index and freezes the shift on the last one, which repeats that
// state and keeps the hardware index aligned with the model's t % 2**WIDTH.
// WIDTH is at least 2: a width-1 LFSR has no feedback polynomial, which is the
// bound get_lfsr_seq itself raises on.


module lfsr #(
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
            ERROR_lfsr_WIDTH_must_be_at_least_2 u_bad ();
        end
    endgenerate

    reg  [WIDTH-1:0] cnt;
    wire             feedback = ^(o_rand & TAPS);
    wire             hold     = &cnt;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt    <= {WIDTH{1'b0}};
            o_rand <= SEED;
        end else if (i_en) begin
            cnt <= cnt + {{(WIDTH-1){1'b0}}, 1'b1};
            if (!hold) begin
                o_rand <= {feedback, o_rand[WIDTH-1:1]};
            end
        end
    end
endmodule
`default_nettype wire
