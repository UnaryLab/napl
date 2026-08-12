`timescale 1ns/1ps
`default_nettype none
// Sample-and-hold re-emitter equivalent to napl.sim.operation.sample_hold; pp_delay=0.
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


module sample_hold #(
    parameter integer WIDTH   = 8,   // sequence length is 2**WIDTH; tb overrides via `GEN_WIDTH
    parameter integer TRIGGER = 128  // one-based latch timestep k, a power of two; tb overrides via `GEN_TRIGGER
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,   // input spike stream
    output wire o_output   // pass-through before the trigger, re-emission after
);
    // Fixed-point fraction shared with the encode sub-instance and its ROM.
    localparam integer FRAC  = WIDTH + 1;
    // count_k / TRIGGER as a FRAC-bit probability is held_count << SHIFT, exact
    // because TRIGGER is a power of two (SHIFT = FRAC - log2(TRIGGER) >= 1).
    localparam integer SHIFT = FRAC - $clog2(TRIGGER);

    // One-based timestep is t = t_idx + 1; t_idx saturates so it never wraps in a run.
    reg [WIDTH-1:0] t_idx;
    // Latched decode count over the first TRIGGER input spikes.
    reg [WIDTH:0]   held_count;
    // Whether the estimate has been latched in the current run.
    reg             latched;

    wire [WIDTH:0]  dec_count;
    wire [WIDTH:0]  t_cur     = {1'b0, t_idx} + {{WIDTH{1'b0}}, 1'b1};
    wire            reemit    = (t_cur >= TRIGGER[WIDTH:0]);
    // At the trigger cycle latched is still low, so the live count is used; after it
    // the frozen register drives, matching the model's latch-then-encode order.
    wire [WIDTH:0]  held_src  = latched ? held_count : dec_count;
    wire [FRAC:0]   enc_in    = {1'b0, held_src} << SHIFT;
    // Holding the encoder in reset until the trigger keeps its sequence index at 0
    // on the trigger cycle; it free-runs thereafter.
    wire            enc_rst_n = i_rst_n & reemit;
    wire            enc_spike;

    decode #(.WIDTH(WIDTH)) u_decode (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input       (i_input),
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
            t_idx      <= {WIDTH{1'b0}};
            held_count <= {(WIDTH+1){1'b0}};
            latched    <= 1'b0;
        end else begin
            if (t_idx != {WIDTH{1'b1}})
                t_idx <= t_idx + {{(WIDTH-1){1'b0}}, 1'b1};
            if (reemit && !latched) begin
                held_count <= dec_count;
                latched    <= 1'b1;
            end
        end
    end
endmodule
`default_nettype wire
