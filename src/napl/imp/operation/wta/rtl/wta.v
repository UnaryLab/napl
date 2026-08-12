`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// wta -- temporal winner-take-all over ENTRY stacked streams.
//
// RTL counterpart of napl.sim.operation.wta (src/napl/sim/operation/wta.py).
// Temporal streams start high and fall once; the earliest falling edge wins.
// The module emits one spike on that edge and stays silent afterwards, and
// simultaneous earliest edges merge into that one spike. The circuit is the
// same for every encoding, so there is no polarity variant.
//
//   falling   = previous & ~i_input          // per stacked stream
//   o_output  = fired ? 0 : |falling
//   fired'    = fired | o_output
//   previous' = i_input
//
// i_input packs the ENTRY streams the Python model stacks along `dim`, stream 0
// in bit 0. The output is combinational over the current input and the held
// state, so pp_delay is 0; each posedge i_clk advances one forward() timestep.
// i_rst_n (active-low) maps to the Python reset(): previous <- all ones, so a
// zero first timestep still reads as a falling edge, and fired <- 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=wta
//==============================================================================


module wta #(
    parameter integer ENTRY = 3      // # stacked streams; tb overrides via `GEN_ENTRY
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire [ENTRY-1:0] i_input,   // stacked temporal spike streams
    output wire             o_output   // winner spike
);
    reg  [ENTRY-1:0] previous;
    reg              fired;
    wire [ENTRY-1:0] falling;
    wire             candidate;

    assign falling = previous & ~i_input;
    assign candidate = |falling;
    assign o_output = fired ? 1'b0 : candidate;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            previous <= {ENTRY{1'b1}};
            fired    <= 1'b0;
        end else begin
            previous <= i_input;
            fired    <= fired | o_output;
        end
    end
endmodule
`default_nettype wire
