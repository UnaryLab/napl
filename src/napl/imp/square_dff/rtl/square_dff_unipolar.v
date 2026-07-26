`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// square_dff_unipolar -- unary/stochastic-computing square, unipolar.
//
// RTL counterpart of napl.sim.operation.square_dff with config polarity='unipolar'
// (src/napl/sim/operation/square_dff.py).  Squares a single spike stream by ANDing it
// with a depth-DEPTH delayed copy of itself (a D flip-flop chain), per uGEMM.
//
//   out = i_input & dff(i_input)        (AND)
//
// The delay line holds the previous DEPTH inputs; the model's internal dff
// reset() initializes every cell to 0. The output is combinational from the
// current input and the oldest cell, so the input->output latency is 0 cycles;
// the delay line advances on posedge i_clk.
//
// DEPTH is a Verilog parameter inherited from the Python model's config['depth']
// (consumed by the embedded dff): the testbench overrides it with `GEN_DEPTH
// (emitted by gen/gen_square_dff.py from the same config test_square_dff.py
// uses), so the verified hardware always tracks the simulator. The default here
// is only a standalone-elaboration fallback.
//
// Reference: uGEMM: Unary Computing (Architecture) for GEMM Applications.
// Verify from src/napl/imp/: conda run -n napl make test OP=square_dff
//==============================================================================
module square_dff_unipolar #(
    parameter integer DEPTH = 1   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,    // sample clock; one tick == one Python forward()
    input  wire i_rst_n,  // active-low reset -> Python reset() (clears delay reg)
    input  wire i_input,     // input spike stream
    output wire o_out     // squared product spike
);
    // depth-DEPTH delay line: in_d[0] is the oldest cell (the delayed copy used
    // this cycle); in_d[DEPTH-1] holds the most recently written input.
    reg [DEPTH-1:0] in_d;

    genvar index;
    generate
        for (index = 0; index < DEPTH; index = index + 1) begin : g_register
            if (index == DEPTH-1) begin : g_tail
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        in_d[index] <= 1'b0;
                    else
                        in_d[index] <= i_input;
                end
            end else begin : g_body
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        in_d[index] <= 1'b0;
                    else
                        in_d[index] <= in_d[index+1];
                end
            end
        end
    endgenerate

    // Combinational product of the current input and its oldest delayed copy.
    assign o_out = i_input & in_d[0];
endmodule
`default_nettype wire
