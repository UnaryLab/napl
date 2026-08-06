`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.avgpool2d over LANES pooled output positions.
// Each lane is one scalar window circuit: accumulate the KERNEL_AREA window
// popcount in units of 1/KERNEL_AREA, emit a spike at or above KERNEL_AREA, and
// subtract KERNEL_AREA on the emit. That is add_any_unipolar with
// SCALE = ENTRY = KERNEL_AREA, so the operation-layer circuit is instantiated
// rather than rebuilt. Generated parameters mirror Python.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every lane accumulator to match reset().
//
// Unpadded windows only: the module drives whole KERNEL_AREA windows of real
// input spikes.


module avgpool2d #(
    parameter integer KERNEL_AREA = 4,   // pooling window elements;  tb overrides via `GEN_KERNEL_AREA
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

    // add_any_unipolar holds A = 2*acc and fires at 2*SCALE, so A stays below
    // 2 * KERNEL_AREA before a fire; one timestep adds at most 2*ENTRY, another
    // 2 * KERNEL_AREA. WIDTH must let ACC_HI = 2**WIDTH - 2 cover that peak.
    localparam integer ACC_WIDTH = clog2(4 * KERNEL_AREA + 2);

    genvar lane;

    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            add_any_unipolar #(
                .SCALE (KERNEL_AREA),
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
