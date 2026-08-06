`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.avgpool2d over LANES pooled output positions.
// Each lane is one scalar window circuit: accumulate the KERNEL_AREA window
// popcount in units of 1/DIVISOR, emit a spike at or above DIVISOR, and subtract
// DIVISOR on the emit. That is add_any_unipolar with SCALE = DIVISOR and
// ENTRY = KERNEL_AREA, so the operation-layer circuit is instantiated rather than
// rebuilt. Generated parameters mirror Python.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every lane accumulator to match reset().
//
// Unpadded windows only: the module drives whole KERNEL_AREA windows of real
// input spikes. The bipolar zero pad of the Python model is a lane-level input
// source and is not part of this circuit.
// DIVISOR >= KERNEL_AREA only, which the generate guard below enforces at
// elaboration; mapping.yaml carries the same restriction.


module avgpool2d #(
    parameter integer KERNEL_AREA = 4,   // pooling window elements;  tb overrides via `GEN_KERNEL_AREA
    parameter integer DIVISOR     = 4,   // window mean divisor;      tb overrides via `GEN_DIVISOR
    parameter integer LANES       = 1    // pooled output positions;  tb overrides via `GEN_LANES
) (
    input  wire                          i_clk,
    input  wire                          i_rst_n,
    input  wire [LANES*KERNEL_AREA-1:0]  i_input_spike,   // lane l occupies bits [l*KERNEL_AREA +: KERNEL_AREA]
    output wire [LANES-1:0]              o_out
);


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    // The lane accumulator stays below DIVISOR + KERNEL_AREA, which holds while
    // DIVISOR >= KERNEL_AREA (a window mean of at most one spike per timestep).
    // add_any_unipolar keeps 2*acc, so size its WIDTH for twice that bound.
    localparam integer ACC_WIDTH = clog2(2 * (DIVISOR + KERNEL_AREA) + 2);

    genvar lane;
    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when the ACC_WIDTH bound above does not hold.
    generate
        if (DIVISOR < KERNEL_AREA) begin : g_bad_divisor
            ERROR_avgpool2d_DIVISOR_must_be_at_least_KERNEL_AREA u_bad ();
        end
    endgenerate

    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            add_any_unipolar #(
                .SCALE (DIVISOR),
                .WIDTH (ACC_WIDTH),
                .ENTRY (KERNEL_AREA)
            ) u_window (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input (i_input_spike[lane*KERNEL_AREA +: KERNEL_AREA]),
                .o_out   (o_out[lane])
            );
        end
    endgenerate
endmodule
`default_nettype wire
