`timescale 1ns/1ps
`default_nettype none
// napl.sim.operation.clamp_comp_dyn: two chained stream comparators bound a
// rate-coded stream to the band the rate-coded streams i_lo and i_hi carry.
// The maximum selector places the floor and the minimum selector places the
// ceiling on its result, which is the max/min composition the Python model
// builds:
//   floor_out = max(i_input, i_lo), o_output = min(floor_out, i_hi).
// Each selector routes one of the two streams it is given, so the output spikes
// are input or bound spikes and no accumulator carries an overshoot.
// One bare module serves both polarities: selection compares spike counts, the
// bipolar value 2p - 1 grows with the rate p, so the same comparison orders both
// polarities and no band mapping is applied. Nothing here is configuration
// dependent, so the module carries no parameters.
// The selectors' o_index outputs report which stream was routed; clamp_comp_dyn
// reads only the routed spike, so both are left unread.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears both selectors' counters and latches, matching reset().


module clamp_comp_dyn (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // rate-coded input spike
    input  wire i_lo,           // rate-coded lower-bound spike
    input  wire i_hi,           // rate-coded upper-bound spike
    output wire o_output        // rate-coded clamped output spike
);
    wire floor_out;
    /* verilator lint_off UNUSEDSIGNAL */
    wire floor_index;
    wire ceiling_index;
    /* verilator lint_on UNUSEDSIGNAL */

    max u_floor_stage (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_input_0 (i_input),
        .i_input_1 (i_lo),
        .o_output  (floor_out),
        .o_index   (floor_index)
    );

    min u_ceiling_stage (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_input_0 (floor_out),
        .i_input_1 (i_hi),
        .o_output  (o_output),
        .o_index   (ceiling_index)
    );
endmodule
`default_nettype wire
