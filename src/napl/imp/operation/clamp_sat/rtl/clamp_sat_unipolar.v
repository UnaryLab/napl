`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.clamp_sat: the same spikes read as the bipolar values
// 2p - 1, so the unipolar band [LO, HI] in grid units of 2^-FRACWIDTH is the
// bipolar band [2*LO - 1, 2*HI - 1] on the bipolar circuit, which this module
// instantiates with that mapped band rather than restating its stages.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the instantiated circuit to match reset().
// Parameters mirror Python, and the tb overrides them from vec/clamp_sat_params.vh.


module clamp_sat_unipolar #(
    parameter integer LO        = 64,   // config['lo'] * 2**FRACWIDTH; tb overrides via `GEN_LO_UNI
    parameter integer HI        = 192,  // config['hi'] * 2**FRACWIDTH; tb overrides via `GEN_HI_UNI
    parameter integer FRACWIDTH = 8,    // config['fracwidth'];         tb overrides via `GEN_FRACWIDTH
    parameter C0_DIRVEC = "vec/clamp_sat_c0.hex",  // Sobol direction vectors of dimension dim
    parameter C1_DIRVEC = "vec/clamp_sat_c1.hex",  // dimension dim + 1
    parameter C2_DIRVEC = "vec/clamp_sat_c2.hex"   // dimension dim + 2
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // unipolar rate-coded input spike
    output wire o_output        // unipolar rate-coded clamped output spike
);
    localparam integer ONE_F = (1 << FRACWIDTH);  // the value 1 in grid units

    clamp_sat_bipolar #(
        .LO        (2 * LO - ONE_F),
        .HI        (2 * HI - ONE_F),
        .FRACWIDTH (FRACWIDTH),
        .C0_DIRVEC (C0_DIRVEC),
        .C1_DIRVEC (C1_DIRVEC),
        .C2_DIRVEC (C2_DIRVEC)
    ) u_band (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input),
        .o_output (o_output)
    );
endmodule
`default_nettype wire
