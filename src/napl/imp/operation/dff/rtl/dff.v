`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// dff -- depth-DEPTH bit-serial delay line (a FIFO of D flip-flops, stateful).
//
// RTL counterpart of napl.sim.operation.dff (src/napl/sim/operation/dff.py). One input
// spike per cycle; the output is the spike that entered DEPTH cycles earlier, so
// the input->output latency is DEPTH (== pp_delay; pp_delay=1 at the default
// depth=1, where the module degenerates to a single D flip-flop).
//
// Reset (active-low i_rst_n) reproduces the Python reset() state EXACTLY: the
// FIFO is initialized to ALL ZEROS (unlike shiftreg's i%2 pattern), so for the
// first DEPTH cycles the output replays zeros, then the DEPTH-cycle-delayed
// inputs follow.
//
// dff has no polarity variants (polarity_required=False), so the module is the
// bare op name with no _unipolar/_bipolar postfix.
//
// DEPTH is a Verilog parameter inherited from the Python model's config['depth']:
// the testbench overrides it with `GEN_DEPTH (emitted by gen/gen_dff.py from the
// same config test_dff.py uses), so the verified hardware always tracks the
// simulator. The default here is only a standalone-elaboration fallback.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=dff
//==============================================================================
module dff #(
    parameter integer DEPTH = 1   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_out     // delayed spike stream (DEPTH cycles old)
);
    // reg_q[0] is the oldest cell (the one read out this cycle); reg_q[DEPTH-1]
    // is the most recently written. Each posedge: emit reg_q[0], shift left, and
    // load i_input into the tail. (At DEPTH=1 this is a single D flip-flop.)
    reg [DEPTH-1:0] reg_q;

    genvar index;
    generate
        for (index = 0; index < DEPTH; index = index + 1) begin : g_register
            if (index == DEPTH-1) begin : g_tail
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        reg_q[index] <= 1'b0;
                    else
                        reg_q[index] <= i_input;
                end
            end else begin : g_body
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        reg_q[index] <= 1'b0;
                    else
                        reg_q[index] <= reg_q[index+1];
                end
            end
        end
    endgenerate

    // Output is always the oldest cell.
    assign o_out = reg_q[0];
endmodule
`default_nettype wire
