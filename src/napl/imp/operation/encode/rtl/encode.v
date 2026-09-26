`timescale 1ns/1ps
`default_nettype none
// Stream encoder equivalent to napl.sim.operation.encode; pp_delay=0.
// An online number-sequence generator produces q[t] and the encoded probability
// is compared against it with a strict greater-than, matching the model's
// s_t = 1{p > q[(t-1) % L]}.
// Polarity only selects how p is derived from the input value (p = x unipolar,
// p = (x+1)/2 bipolar), so one bare module covers both.
// GENERATOR selects the sequence circuit at elaboration, one module per branch on
// a shared o_rand socket, mirroring gen_num_seq's generator key:
//   0 = sobol   (`sobol`, `rc`, `rate`), reads DIRVEC_FILE
//   1 = lfsr    (`lfsr`),                reads TAPS and SEED
//   2 = lfsr_ext(`lfsr_ext`),            reads TAPS and SEED
//   3 = temporalseq (`tc`, `temporal`),  reads neither
// Every branch is registered on the same clock and its value is compared in the
// arrival cycle, so pp_delay is 0 in every mode. TAPS, SEED, and DIRVEC_FILE are
// carried for all modes and ignored by the modes that do not read them; their
// defaults are placeholders, not a working table for any particular width.
// Generated WIDTH mirrors Python self.width = ceil(log2(config['timestep']));
// FRAC is the fixed-point fraction of i_input, at least WIDTH bits, and the
// WIDTH-bit sequence value is left-aligned onto that grid.
// A DIRVEC_FILE override is truncated to the 24-character default's width, so
// keep override paths at 24 characters or fewer.


module encode #(
    parameter integer WIDTH = 8,  // sequence length is 2**WIDTH; tb overrides via `GEN_WIDTH
    parameter integer FRAC  = 9,  // fractional bits of i_input, FRAC >= WIDTH; tb overrides via `GEN_FRAC
    parameter integer GENERATOR = 0,   // sequence circuit: 0 sobol, 1 lfsr, 2 lfsr_ext, 3 temporalseq
    parameter [WIDTH-1:0] TAPS = 1,    // lfsr/lfsr_ext feedback tap mask
    parameter [WIDTH-1:0] SEED = 1,    // lfsr/lfsr_ext post-reset state
    parameter DIRVEC_FILE = "vec/encode_dirvec_d1.hex"  // Sobol direction vectors, relative to the simulation cwd
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire [FRAC:0]    i_input,   // encoded probability p in [0, 2**FRAC]
    output wire             o_spike
);
    generate
        if (FRAC < WIDTH) begin : g_bad_frac
            // An unresolvable module reference makes iverilog fail the build when
            // the input grid is coarser than the sequence grid.
            ERROR_encode_FRAC_too_small_for_WIDTH u_bad ();
        end
    endgenerate

    wire [WIDTH-1:0] num_seq;
    wire [FRAC:0]    num_seq_code;

    generate
        if (GENERATOR == 0) begin : g_sobol
            sobol #(
                .WIDTH       (WIDTH),
                .DIRVEC_FILE (DIRVEC_FILE)
            ) u_seq (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                // sobol gates its counter and value update on i_en; the constant
                // high here advances the sequence on every cycle.
                .i_en    (1'b1),
                .o_rand  (num_seq)
            );
        end else if (GENERATOR == 1) begin : g_lfsr
            lfsr #(
                .WIDTH (WIDTH),
                .TAPS  (TAPS),
                .SEED  (SEED)
            ) u_seq (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                // lfsr gates its shift and index count on i_en; the constant high
                // here advances the sequence on every cycle.
                .i_en    (1'b1),
                .o_rand  (num_seq)
            );
        end else if (GENERATOR == 2) begin : g_lfsr_ext
            lfsr_ext #(
                .WIDTH (WIDTH),
                .TAPS  (TAPS),
                .SEED  (SEED)
            ) u_seq (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                // lfsr_ext gates its shift on i_en; the constant high here
                // advances the sequence on every cycle.
                .i_en    (1'b1),
                .o_rand  (num_seq)
            );
        end else if (GENERATOR == 3) begin : g_temporalseq
            temporalseq #(
                .WIDTH (WIDTH)
            ) u_seq (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                // temporalseq gates its counter on i_en; the constant high here
                // advances the sequence on every cycle.
                .i_en    (1'b1),
                .o_rand  (num_seq)
            );
        end else begin : g_bad_generator
            // An unresolvable module reference makes iverilog fail the build when
            // GENERATOR names no implemented sequence circuit, so an unsupported
            // generator cannot silently synthesize a different sequence.
            ERROR_encode_GENERATOR_must_select_a_supported_sequence u_bad ();
        end
    endgenerate

    // num_seq counts units of 2**-WIDTH; i_input counts units of 2**-FRAC.
    assign num_seq_code = {{(FRAC+1-WIDTH){1'b0}}, num_seq} << (FRAC - WIDTH);
    assign o_spike = (i_input > num_seq_code) ? 1'b1 : 1'b0;
endmodule
`default_nettype wire
