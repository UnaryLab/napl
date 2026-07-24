`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// shiftreg -- depth-DEPTH bit-serial shift register (stateful).
//
// RTL counterpart of napl.sim.operation.shiftreg (src/napl/sim/operation/shiftreg.py).
// One input spike per cycle; the output is the spike that entered DEPTH cycles
// earlier, so the input->output latency is DEPTH (== pp_delay).
//
// Reset (active-low i_rst_n) reproduces the Python reset() state EXACTLY:
// reg[i] = i % 2 (an alternating 0,1,0,1,... pattern), NOT all-zeros. The model
// reads the oldest cell first, so for the first DEPTH cycles the output replays
// this reset pattern (reg[0], reg[1], ...), then the delayed inputs follow.
//
// shiftreg has no polarity variants (polarity_required=False), so the module is
// the bare op name with no _unipolar/_bipolar postfix.
//
// DEPTH is a Verilog parameter inherited from the Python model's config['depth']:
// the testbench overrides it with `GEN_DEPTH (emitted by gen/gen_shiftreg.py from
// the same config test_shiftreg.py uses), so the verified hardware always tracks
// the simulator. The default here is only a standalone-elaboration fallback.
//==============================================================================
module shiftreg #(
    parameter integer DEPTH = 2   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_in,     // input spike stream
    output wire o_out     // delayed spike stream (DEPTH cycles old)
);
    // reg_q[0] is the oldest cell (the one read out this cycle); reg_q[DEPTH-1]
    // is the most recently written. Each posedge: emit reg_q[0], shift left, and
    // load i_in into the tail.
    reg [DEPTH-1:0] reg_q;

    integer i;
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            // Reload the model's reset() state: reg[i] = i % 2.
            for (i = 0; i < DEPTH; i = i + 1)
                reg_q[i] <= i[0];
        end else begin
            // Shift: drop the oldest (reg_q[0]), slide the rest down, append i_in.
            for (i = 0; i < DEPTH-1; i = i + 1)
                reg_q[i] <= reg_q[i+1];
            reg_q[DEPTH-1] <= i_in;
        end
    end

    // Output is always the oldest cell.
    assign o_out = reg_q[0];
endmodule
`default_nettype wire
