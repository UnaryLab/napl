`timescale 1ns/1ps
`default_nettype none
// Sample-and-hold re-emitter equivalent to napl.sim.operation.encode_hold; pp_delay=0.
// Phases keyed on the current one-based timestep t and the trigger k = TRIGGER:
//   t <  k : pass the live input spike through.
//   t == k : latch the decoded estimate of the first k input spikes.
//   t >= k : re-emit a fresh Sobol-encoded stream of the frozen estimate.
// Composition: a `decode` counter accumulates the input, and its count at t = k is
// the frozen estimate count_k. An `encode` re-encoder, held in reset until t = k so
// its sequence index is 0 on the trigger cycle and advances thereafter, emits that
// estimate. The re-encode probability is count_k / TRIGGER for BOTH polarities
// (unipolar held = c/k with p = held; bipolar held = 2c/k - 1 with p = (held+1)/2 =
// c/k), so one bare module covers both, matching decode and encode.
// Restriction: TRIGGER must be a power of two, so count_k / TRIGGER is exact in the
// model's float decode and the left shift below reproduces it bit for bit. The sim
// supports any trigger, but only a power-of-two trigger has a bit-exact circuit.
// Generated WIDTH = ceil(log2(timestep)); TRIGGER = trigger_timestep.


module encode_hold #(
    parameter integer WIDTH   = 8,   // sequence length is 2**WIDTH; tb overrides via `GEN_WIDTH
    parameter integer TRIGGER = 128  // one-based latch timestep k, a power of two; tb overrides via `GEN_TRIGGER
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,   // input spike stream
    output wire o_output   // pass-through before the trigger, re-emission after
);
    // ---- ceil(log2(x)) constant function (Verilog-2001) ----


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    // Fixed-point fraction shared with the encode sub-instance.
    localparam integer FRAC  = WIDTH + 1;
    // count_k / TRIGGER as a FRAC-bit probability is count_k << SHIFT, exact
    // because TRIGGER is a power of two (SHIFT = FRAC - log2(TRIGGER) >= 1).
    localparam integer SHIFT = FRAC - clog2(TRIGGER);
    // The timestep counter only has to reach the trigger, and the count only
    // counts the first TRIGGER spikes, so both are sized from TRIGGER.
    localparam integer TW    = (clog2(TRIGGER) > 0) ? clog2(TRIGGER) : 1;
    localparam integer T_LAST_INT = TRIGGER - 1;
    localparam [TW-1:0] T_LAST = T_LAST_INT[TW-1:0];
    localparam [TW-1:0] T_ONE  = 1;

    // Zero-based timestep t - 1; it may wrap once latched, which then dominates.
    reg  [TW-1:0] t_idx;
    // Whether the estimate has been latched in the current run.
    reg           latched;

    wire          reemit    = latched | (t_idx == T_LAST);
    // decode stops counting once latched, so from the trigger cycle on its count
    // is count_k: live on the trigger cycle, frozen after it.
    wire [TW:0]   dec_count;
    wire [FRAC:0] enc_in    = {{(FRAC-TW){1'b0}}, dec_count} << SHIFT;
    // Holding the encoder in reset until the trigger keeps its sequence index at 0
    // on the trigger cycle; it free-runs thereafter.
    wire          enc_rst_n = i_rst_n & reemit;
    wire          enc_spike;

    decode #(.WIDTH(TW)) u_decode (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input       (i_input & ~latched),
        .o_spike_count (dec_count)
    );

    encode #(.WIDTH(WIDTH), .FRAC(FRAC)) u_encode (
        .i_clk   (i_clk),
        .i_rst_n (enc_rst_n),
        .i_input (enc_in),
        .o_spike (enc_spike)
    );

    assign o_output = reemit ? enc_spike : i_input;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            t_idx   <= {TW{1'b0}};
            latched <= 1'b0;
        end else begin
            t_idx   <= t_idx + T_ONE;
            latched <= reemit;
        end
    end
endmodule
`default_nettype wire
