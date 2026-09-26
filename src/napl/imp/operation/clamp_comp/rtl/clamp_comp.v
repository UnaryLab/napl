`timescale 1ns/1ps
`default_nettype none
// napl.sim.operation.clamp_comp: two chained stream comparators bound a rate-coded
// stream to the fixed band [lo, hi]. The maximum selector places the floor and the
// minimum selector places the ceiling on its result, which is the max/min
// composition the Python model builds:
//   floor_out = max(i_input, lo_spike), o_output = min(floor_out, hi_spike).
// Each selector routes one of the two streams it is given, so the output spikes are
// input or bound spikes and no accumulator carries an overshoot.
// The two bounds are encoded here rather than supplied: one encode cell per bound
// drives a Sobol stream of period 2**SEQ_W on consecutive dimensions, whose
// direction vectors come from LO_DIRVEC and HI_DIRVEC. A bound is carried as its
// probability code in units of 2**-SEQ_W, which is the threshold the model compares
// against the same sequence: the code of a unipolar bound b is b * 2**SEQ_W and of a
// bipolar bound b is (b + 1) / 2 * 2**SEQ_W.
// One bare module serves both polarities: selection compares spike counts, the
// bipolar value 2p - 1 grows with the rate p, so the same comparison orders both
// polarities, and the polarity reaches this circuit only through the two bound
// codes, which are parameters rather than logic.
// The selectors' o_index outputs report which stream was routed; clamp_comp reads
// only the routed spike, so both are left unread.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the selectors' counters and latches and restarts both
// bound sequences at index 0, matching reset().
// Parameters mirror Python, and the tb overrides them from vec/clamp_comp_params.vh.


module clamp_comp #(
    parameter integer SEQ_W   = 9,    // log2 of self.constant_len; tb overrides via `GEN_SEQ_W
    parameter integer LO_CODE = 128,  // lo probability * 2**SEQ_W; tb overrides via `GEN_LO_CODE_BI
    parameter integer HI_CODE = 384,  // hi probability * 2**SEQ_W; tb overrides via `GEN_HI_CODE_BI
    parameter LO_DIRVEC = "vec/clamp_comp_lo.hex",  // Sobol direction vectors of dimension dim
    parameter HI_DIRVEC = "vec/clamp_comp_hi.hex"   // dimension dim + 1
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // rate-coded input spike
    output wire o_output        // rate-coded clamped output spike
);
    // The bound codes reach the encoder as unsigned probabilities on the
    // 2**-SEQ_W grid, so each needs the SEQ_W + 1 bits that hold 2**SEQ_W.
    localparam [SEQ_W:0] LO_THRESHOLD = LO_CODE[SEQ_W:0];
    localparam [SEQ_W:0] HI_THRESHOLD = HI_CODE[SEQ_W:0];

    wire lo_spike;
    wire hi_spike;
    wire floor_out;
    /* verilator lint_off UNUSEDSIGNAL */
    wire floor_index;
    wire ceiling_index;
    /* verilator lint_on UNUSEDSIGNAL */

    encode #(
        .WIDTH       (SEQ_W),
        .FRAC        (SEQ_W),
        .GENERATOR   (0),
        .DIRVEC_FILE (LO_DIRVEC)
    ) u_lo_bound (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (LO_THRESHOLD),
        .o_spike (lo_spike)
    );

    encode #(
        .WIDTH       (SEQ_W),
        .FRAC        (SEQ_W),
        .GENERATOR   (0),
        .DIRVEC_FILE (HI_DIRVEC)
    ) u_hi_bound (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (HI_THRESHOLD),
        .o_spike (hi_spike)
    );

    max u_floor_stage (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_input_0 (i_input),
        .i_input_1 (lo_spike),
        .o_output  (floor_out),
        .o_index   (floor_index)
    );

    min u_ceiling_stage (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_input_0 (floor_out),
        .i_input_1 (hi_spike),
        .o_output  (o_output),
        .o_index   (ceiling_index)
    );
endmodule
`default_nettype wire
