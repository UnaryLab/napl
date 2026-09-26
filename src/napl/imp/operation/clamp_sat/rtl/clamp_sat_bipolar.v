`timescale 1ns/1ps
`default_nettype none
// Bipolar napl.sim.operation.clamp_sat: three chained unit-scale saturating adders
// bound a rate-coded stream to the band [LO, HI], in grid units of 2^-FRACWIDTH.
// With sat(v) the rate saturation of a bipolar add_scale to [-1, 1], the stages are
//   u = sat(v - (1 + lo)), w = sat(u + (2 + lo - hi)), y = sat(w + (hi - 1)),
// so the first stage's low saturation places the floor, the second stage's high
// saturation places the ceiling, and the third shifts the band back.
// Each stage is a three-entry add_scale_bipolar carrying the stream, one constant
// stream, and one fixed rail: absent (never spiking) on the floor and shift stages,
// always spiking on the ceiling stage. That split leaves stream constants -lo,
// 1 + lo - hi, and hi, each of magnitude at most 1 for every legal band.
// The three constants ride Sobol streams of period 2**(FRACWIDTH+1) that one encode
// cell each generates, on consecutive Sobol dimensions whose direction vectors come
// from C0_DIRVEC, C1_DIRVEC, and C2_DIRVEC. A constant of value c is the encoded
// probability (c + 1) / 2, whose code in units of 2^-(FRACWIDTH+1) is the integer
// below; the band bounds keep every code inside [0, 2**(FRACWIDTH+1)].
// Restriction: FRACWIDTH + 2 <= 31, so the 2*ONE_F elaboration constants stay
// inside the 32-bit signed range.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the stage accumulators and the sequence generators to
// match reset(). Parameters mirror Python, and the tb overrides them from
// vec/clamp_sat_params.vh.


module clamp_sat_bipolar #(
    parameter integer LO        = -128,  // config['lo'] * 2**FRACWIDTH; tb overrides via `GEN_LO_BI
    parameter integer HI        =  128,  // config['hi'] * 2**FRACWIDTH; tb overrides via `GEN_HI_BI
    parameter integer FRACWIDTH = 8,     // config['fracwidth'];        tb overrides via `GEN_FRACWIDTH
    parameter C0_DIRVEC = "vec/clamp_sat_c0.hex",  // Sobol direction vectors of dimension dim
    parameter C1_DIRVEC = "vec/clamp_sat_c1.hex",  // dimension dim + 1
    parameter C2_DIRVEC = "vec/clamp_sat_c2.hex"   // dimension dim + 2
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // bipolar rate-coded input spike
    output wire o_output        // bipolar rate-coded clamped output spike
);
    // ---- derived sizing (all magic constants trace to the parameters) ----

    localparam integer SEQ_W = FRACWIDTH + 1;    // constant-stream period is 2**SEQ_W
    localparam integer ONE_F = (1 << FRACWIDTH); // the value 1 in grid units

    // Stage shape: the stream, its constant, and the fixed rail, on unit scale.
    // Four integer bits reach the accuracy a wider accumulator holds.
    localparam integer STAGE_SCALE = 1;
    localparam integer STAGE_WIDTH = 4;
    localparam integer STAGE_ENTRY = 3;

    // Elaboration-time guard: an unresolvable module reference makes iverilog fail
    // the build when FRACWIDTH pushes the 2*ONE_F term of C1_CODE, and the 2*LO
    // term clamp_sat_unipolar maps its band with, out of the 32-bit signed range.
    generate
        if (FRACWIDTH + 2 > 31) begin : g_bad_sizing
            ERROR_clamp_sat_FRACWIDTH_overflows_32_bit_constants u_bad ();
        end
    endgenerate

    // Probability codes of the stream constants, in units of 2^-(FRACWIDTH+1).
    localparam integer C0_CODE = ONE_F - LO;              // constant -lo
    localparam integer C1_CODE = 2 * ONE_F + LO - HI;     // constant 1 + lo - hi
    localparam integer C2_CODE = ONE_F + HI;              // constant hi

    wire c0_spike;
    wire c1_spike;
    wire c2_spike;
    wire floor_out;
    wire ceiling_out;

    encode #(
        .WIDTH       (SEQ_W),
        .FRAC        (SEQ_W),
        .GENERATOR   (0),
        .DIRVEC_FILE (C0_DIRVEC)
    ) u_constant_0 (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (C0_CODE[SEQ_W:0]),
        .o_spike (c0_spike)
    );

    encode #(
        .WIDTH       (SEQ_W),
        .FRAC        (SEQ_W),
        .GENERATOR   (0),
        .DIRVEC_FILE (C1_DIRVEC)
    ) u_constant_1 (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (C1_CODE[SEQ_W:0]),
        .o_spike (c1_spike)
    );

    encode #(
        .WIDTH       (SEQ_W),
        .FRAC        (SEQ_W),
        .GENERATOR   (0),
        .DIRVEC_FILE (C2_DIRVEC)
    ) u_constant_2 (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (C2_CODE[SEQ_W:0]),
        .o_spike (c2_spike)
    );

    // The stage input is a popcount, so the lane order carries no meaning: the
    // top lane is the fixed rail, absent here and present on the ceiling stage.
    add_scale_bipolar #(
        .SCALE (STAGE_SCALE),
        .WIDTH (STAGE_WIDTH),
        .ENTRY (STAGE_ENTRY)
    ) u_floor_stage (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  ({1'b0, c0_spike, i_input}),
        .o_output (floor_out)
    );

    add_scale_bipolar #(
        .SCALE (STAGE_SCALE),
        .WIDTH (STAGE_WIDTH),
        .ENTRY (STAGE_ENTRY)
    ) u_ceiling_stage (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  ({1'b1, c1_spike, floor_out}),
        .o_output (ceiling_out)
    );

    add_scale_bipolar #(
        .SCALE (STAGE_SCALE),
        .WIDTH (STAGE_WIDTH),
        .ENTRY (STAGE_ENTRY)
    ) u_shift_stage (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  ({1'b0, c2_spike, ceiling_out}),
        .o_output (o_output)
    );
endmodule
`default_nettype wire
