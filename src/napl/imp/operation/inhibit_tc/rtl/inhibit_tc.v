`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// inhibit_tc -- temporal-code (race logic) INHIBIT gate with a sticky flag.
//
// RTL counterpart of napl.sim.operation.inhibit_tc (src/napl/sim/operation/inhibit_tc.py).
// No polarity variants and no sizing configuration, so a single bare module.
//
//   set  = i_input_data & ~i_input_inhibit     sampled at each posedge i_clk
//   q    holds 1 once set was high at a clock edge; cleared only by !i_rst_n
//   o_output = i_input_data | q
//
// The model's output is data | (latch | set), and set implies data, so the
// current-cycle set term drops out and the output is data | q from the flop
// holding the earlier timesteps' sets: combinational, pp_delay = 0.
// i_rst_n (active-low) asynchronously clears q and maps to the Python reset():
// latch <- 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=inhibit_tc
//==============================================================================


module inhibit_tc (
    input  wire i_clk,             // one posedge == one Python forward() timestep
    input  wire i_rst_n,           // active-low; maps to Python reset()
    input  wire i_input_data,      // data temporal stream
    input  wire i_input_inhibit,   // inhibiting temporal stream
    output wire o_output           // data spike held high by the inhibition flag
);
    reg q;

    assign o_output = i_input_data | q;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            q <= 1'b0;
        else if (i_input_data & ~i_input_inhibit)
            q <= 1'b1;
    end
endmodule
`default_nettype wire
