`timescale 1ns/1ps
`default_nettype none
// DEPTH-cycle delay equivalent to napl.sim.operation.delay; pp_delay=DEPTH.
// Generated DEPTH mirrors Python. Active-low reset loads the initial pattern
// config['init'] selects: all zeros at INIT=0, and cell j holding j%2 at INIT=1.
// The oldest cell is emitted first.


module delay #(
    parameter integer DEPTH = 1,  // inherited from config['depth']; tb overrides via `GEN_DEPTH
    parameter integer INIT  = 0   // 0 == config['init']=="zero", 1 == "alternate"
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,  // input spike stream
    output wire o_output  // delayed spike stream (DEPTH cycles old)
);
    // reg_q[0] is the oldest cell (the one read out this cycle); reg_q[DEPTH-1]
    // is the most recently written. Each posedge: emit reg_q[0], shift left, and
    // load i_input into the tail. (At DEPTH=1 this is a single D flip-flop.)
    // reg_q[j] holds the Python model's reg[j], so the reset pattern is _fill's.
    reg [DEPTH-1:0] reg_q;

    genvar index;
    generate
        for (index = 0; index < DEPTH; index = index + 1) begin : g_register
            if (index == DEPTH-1) begin : g_tail
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        reg_q[index] <= ((INIT != 0) && ((index % 2) == 1)) ? 1'b1 : 1'b0;
                    else
                        reg_q[index] <= i_input;
                end
            end else begin : g_body
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        reg_q[index] <= ((INIT != 0) && ((index % 2) == 1)) ? 1'b1 : 1'b0;
                    else
                        reg_q[index] <= reg_q[index+1];
                end
            end
        end
    endgenerate

    assign o_output = reg_q[0];
endmodule
`default_nettype wire
