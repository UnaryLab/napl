`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// inhibit -- temporal-code (race logic) INHIBIT gate with a sticky latch.
//
// RTL counterpart of napl.sim.operation.inhibit (src/napl/sim/operation/inhibit.py).
// No polarity variants and no sizing configuration, so a single bare module.
//
//   s_t = s_{t-1} | (i_input_0 & ~i_input_1)      sticky inhibition latch
//   o_out = i_input_0 | s_t
//
// Both inputs are consumed combinationally in their arrival cycle, so the output
// is available the same cycle (pp_delay = 0); the latch only carries state
// forward. i_rst_n (active-low) maps to the Python reset(): s <- 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=inhibit
//==============================================================================


module inhibit (
    input  wire i_clk,       // one posedge == one Python forward() timestep
    input  wire i_rst_n,     // active-low; maps to Python reset()
    input  wire i_input_0,   // data temporal stream
    input  wire i_input_1,   // inhibiting temporal stream
    output wire o_out        // data spike held high by the inhibition latch
);
    reg  latch_q;

    wire blocked = i_input_0 & ~i_input_1;
    wire latch_d = latch_q | blocked;

    assign o_out = i_input_0 | latch_d;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            latch_q <= 1'b0;
        else
            latch_q <= latch_d;
    end
endmodule
`default_nettype wire
