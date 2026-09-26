`timescale 1ns/1ps
`default_nettype none
// napl.sim.operation.clamp_sat_dyn: three chained unit-scale saturating adders bound a
// rate-coded stream to the band the rate-coded streams i_lo and i_hi carry.
// With sat(v) the rate saturation of a bipolar add_scale to [-1, 1], the stages are
//   u = sat(v - (1 + lo)), w = sat(u + (2 + lo - hi)), y = sat(w + (hi - 1)),
// so the first stage's low saturation places the floor, the second stage's high
// saturation places the ceiling, and the third shifts the band back.
// A bound enters a stage directly for +lo or +hi and inverted for -lo or -hi, since
// 1 - spike encodes the negated bipolar value. The remaining addends are fixed rails:
// a lane tied low carries -1 and a lane tied high carries +1. That makes the floor
// stage (v, rail -1, ~lo), the ceiling stage (u, rail +1, rail +1, lo, ~hi), and the
// shift stage (w, rail -1, hi). add_scale centers a bipolar sum by (entry - scale) / 2,
// which is a whole accumulator unit for both the three-entry and the five-entry stage.
// One module serves both polarities: no band is baked here, and a unipolar bound
// stream read as bipolar spikes already carries the value 2p - 1, so the unipolar
// band is clamped by these same stages on the same spikes.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the stage accumulators to match reset().
// WIDTH mirrors the stage intwidth the Python model builds its adders with, and the
// tb overrides it from vec/clamp_sat_dyn_params.vh. add_scale_bipolar carries the
// elaboration guard on it.


module clamp_sat_dyn #(
    parameter integer WIDTH = 4   // stage add_scale intwidth; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // rate-coded input spike
    input  wire i_lo,           // rate-coded lower-bound spike
    input  wire i_hi,           // rate-coded upper-bound spike
    output wire o_output        // rate-coded clamped output spike
);
    // Stage shape: unit scale, with the entry count set by each stage's lane list.
    localparam integer STAGE_SCALE = 1;
    localparam integer FLOOR_ENTRY = 3;
    localparam integer CEIL_ENTRY  = 5;
    localparam integer SHIFT_ENTRY = 3;

    wire floor_out;
    wire ceiling_out;

    // The stage input is a popcount, so the lane order carries no meaning.
    add_scale_bipolar #(
        .SCALE (STAGE_SCALE),
        .WIDTH (WIDTH),
        .ENTRY (FLOOR_ENTRY)
    ) u_floor_stage (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  ({1'b0, ~i_lo, i_input}),
        .o_output (floor_out)
    );

    add_scale_bipolar #(
        .SCALE (STAGE_SCALE),
        .WIDTH (WIDTH),
        .ENTRY (CEIL_ENTRY)
    ) u_ceiling_stage (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  ({1'b1, 1'b1, i_lo, ~i_hi, floor_out}),
        .o_output (ceiling_out)
    );

    add_scale_bipolar #(
        .SCALE (STAGE_SCALE),
        .WIDTH (WIDTH),
        .ENTRY (SHIFT_ENTRY)
    ) u_shift_stage (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  ({1'b0, i_hi, ceiling_out}),
        .o_output (o_output)
    );
endmodule
`default_nettype wire
