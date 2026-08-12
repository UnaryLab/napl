`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// inhibit -- temporal-code (race logic) INHIBIT gate with a sticky latch.
//
// RTL counterpart of napl.sim.operation.inhibit (src/napl/sim/operation/inhibit.py).
// No polarity variants and no sizing configuration, so a single bare module.
//
//   set = i_input_data & ~i_input_inhibit                  set-dominant latch input
//   latch_q holds 1 once set has been high; cleared only by !i_rst_n
//   o_output = i_input_data | latch_q
//
// The latch is level-sensitive and clockless: set propagates into latch_q as soon
// as the inputs arrive, so the output is available in the arrival cycle
// (pp_delay = 0). i_rst_n (active-low) asynchronously clears the latch and maps to
// the Python reset(): s <- 0.
//
// This deliberately deviates from the project's lint-clean rule that forbids
// inferred latches: the missing else in the always @* block infers the latch on
// purpose, because this is asynchronous race logic rather than clocked datapath.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=inhibit
//==============================================================================


module inhibit (
    input  wire i_clk,             // unused: clockless race logic, kept for interface consistency
    input  wire i_rst_n,           // active-low; maps to Python reset()
    input  wire i_input_data,      // data temporal stream
    input  wire i_input_inhibit,   // inhibiting temporal stream
    output wire o_output           // data spike held high by the inhibition latch
);
    reg  latch_q;

    wire set = i_input_data & ~i_input_inhibit;

    assign o_output = i_input_data | latch_q;

    always @* begin
        if (!i_rst_n)
            latch_q = 1'b0;
        else if (set)
            latch_q = 1'b1;
    end
endmodule
`default_nettype wire
