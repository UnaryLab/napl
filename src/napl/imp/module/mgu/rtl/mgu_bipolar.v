`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.mgu (bipolar only) over LANES = hidden_size lanes.
//
// The Python forward() is, term by term:
//   fg_in = fg_ug_tanh(cat(hx_spike, input_spike))   linear, scale 1
//   fg    = fg_sigmoid(fg_in)                        sigmoid_hard
//   fg_hx = fg_hx_mul(fg, hx_value)                  mul_ugemm  (held operand)
//   ng    = ng_ug_tanh(cat(fg_hx, input_spike))      linear, scale 1
//   fg_ng = fg_ng_mul(fg, ng)                        mul_ugemm_sr
//   out   = hy_add(ng + (1-fg_ng) + fg_hx, entry=3)  add_any, scale 1
// Every term is an existing operation- or module-layer circuit, so this file
// instantiates them and adds no scalar logic of its own. The one gate written
// here is the inverter for (1 - fg_ng), which is a bipolar stream negation.
//
// Cross-lane fan-out: both gates are ONE linear_bipolar covering all LANES, not
// a per-lane replication. Their input is IN_FEATURES = LANES + IN_SIZE wide and
// every lane's row reads every lane's feature, which is the cat() in the model.
// The fg gate reads {input_spike, hx_spike} and the n gate reads
// {input_spike, fg_hx}: same width, different source for the low LANES bits.
//
// Held operands and streams: weight and bias arrive as spikes on ports, because
// the model re-encodes both from external tensors every timestep. i_hx_value is
// the fixed-point code of the model's hx_value buffer, held for the whole run,
// because mul_ugemm reads a numeric operand rather than a stream.
//
// Shared model state that this file replicates per lane: mul_ugemm_sr's `head`
// and mul_ugemm's sequence indices are single tensors in Python that broadcast
// over the lanes. Every lane advances the head unconditionally, once per clock,
// and its sequence indices from its own fg spike, so the per-lane copies track
// the model exactly.
//
// Cost: each lane holds one mul_ugemm_sr_bipolar, whose shift register is
// 2**SR_WIDTH flops (64 at the default depth_ismul of 6), so the cell carries
// LANES * 2**SR_WIDTH decorrelation flops before anything else.
//
// Timing, stated rather than fixed: pp_delay = 0, so the longest combinational
// chain, f-linear -> sigmoid_hard -> mul_ugemm -> n-linear -> add_any, settles
// inside one clock, and its through-delay is the sum of two
// linear popcount-and-accumulate stages plus three operation stages. That is a
// deep combinational path and a real Fmax limit at synthesis. It is not fixed
// here: one Python forward() timestep is one posedge, so a pipeline register
// would break that contract and the co-simulation, which has no timing model,
// would not see the difference either way.
//
// Restrictions, mirrored as `requires` in mapping.yaml:
//   - SEQ_WIDTH > SR_WIDTH, the RTL form of the model's
//     timestep > 2 ** depth_ismul guard; the generate guard below enforces it.
//     SEQ_WIDTH = ceil(log2(timestep)) loses the exact timestep, so this is the
//     strongest statement of that restriction available at elaboration.
//   - 2 ** (WIDTH - 1) > IN_FEATURES + bias, enforced at elaboration by the
//     ERROR_linear_WIDTH_too_small_for_ENTRY guard inside linear_bipolar.
//   - WIDTH <= 30, from the add_any_bipolar 32-bit elaboration constants.
// Active-low reset returns every child to its post-reset() Python state.


module mgu_bipolar #(
    parameter integer LANES      = 3,   // hidden units;        tb overrides via `GEN_LANES
    parameter integer IN_SIZE    = 4,   // input features;      tb overrides via `GEN_IN_SIZE
    parameter integer WIDTH      = 10,  // accumulator width;   tb overrides via `GEN_WIDTH
    parameter integer SEQ_WIDTH  = 8,   // ceil(log2(timestep));tb overrides via `GEN_SEQ_WIDTH
    parameter integer SR_WIDTH   = 6,   // depth_ismul;         tb overrides via `GEN_SR_WIDTH
    parameter integer HAS_BIAS_F = 1,   // 1 adds a forget-gate bias spike
    parameter integer HAS_BIAS_N = 1    // 1 adds a new-gate bias spike
) (
    input  wire                             i_clk,
    input  wire                             i_rst_n,
    input  wire [IN_SIZE-1:0]               i_input_spike,
    input  wire [LANES-1:0]                 i_hx_spike,
    // lane l, feature f at [l*(LANES + IN_SIZE) + f]; f < LANES is the hidden
    // half of the cat(), f >= LANES the input half.
    input  wire [LANES*(LANES+IN_SIZE)-1:0] i_weight_f,
    input  wire [LANES-1:0]                 i_bias_f,
    input  wire [LANES*(LANES+IN_SIZE)-1:0] i_weight_n,
    input  wire [LANES-1:0]                 i_bias_n,
    // lane l at [l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1], a code in [0, 2**SEQ_WIDTH]
    input  wire [LANES*(SEQ_WIDTH+1)-1:0]   i_hx_value,
    output wire [LANES-1:0]                 o_out
);


    localparam integer IN_FEATURES = LANES + IN_SIZE;  // cat(hidden, input) width
    // The model builds both gate layers with scale 1, and the output adder with
    // scale 1 over the three addends ng, 1 - fg_ng, and fg_hx.
    localparam integer GATE_SCALE = 1;
    localparam integer OUT_SCALE  = 1;
    localparam integer OUT_ENTRY  = 3;


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when the run cannot outlast the multiplier shift register.
    // timestep > 2**SR_WIDTH with SEQ_WIDTH = ceil(log2(timestep)) gives
    // 2**SEQ_WIDTH >= timestep > 2**SR_WIDTH, hence SEQ_WIDTH > SR_WIDTH.
    generate
        if (SEQ_WIDTH <= SR_WIDTH) begin : g_bad_run_length
            ERROR_mgu_SEQ_WIDTH_must_exceed_SR_WIDTH u_bad ();
        end
    endgenerate

    wire [LANES-1:0]       fg_in;
    wire [LANES-1:0]       fg;
    wire [LANES-1:0]       fg_hx;
    wire [LANES-1:0]       ng;
    wire [LANES-1:0]       fg_ng;
    wire [IN_FEATURES-1:0] fg_gate_in = {i_input_spike, i_hx_spike};
    wire [IN_FEATURES-1:0] ng_gate_in = {i_input_spike, fg_hx};

    linear_bipolar #(
        .IN_FEATURES (IN_FEATURES),
        .LANES       (LANES),
        .WIDTH       (WIDTH),
        .SCALE       (GATE_SCALE),
        .HAS_BIAS    (HAS_BIAS_F)
    ) u_fg_lin (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (fg_gate_in),
        .i_weight      (i_weight_f),
        .i_bias        (i_bias_f),
        .o_out         (fg_in)
    );

    linear_bipolar #(
        .IN_FEATURES (IN_FEATURES),
        .LANES       (LANES),
        .WIDTH       (WIDTH),
        .SCALE       (GATE_SCALE),
        .HAS_BIAS    (HAS_BIAS_N)
    ) u_ng_lin (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (ng_gate_in),
        .i_weight      (i_weight_n),
        .i_bias        (i_bias_n),
        .o_out         (ng)
    );

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            // sigmoid_hard reaches only accumulator states 0 and 1 for a 0/1
            // input, so its fixed internal width matches the model at any
            // configured width; nothing here depends on WIDTH.
            sigmoid_hard u_sig (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input (fg_in[lane]),
                .o_out   (fg[lane])
            );

            mul_ugemm_bipolar #(
                .WIDTH (SEQ_WIDTH)
            ) u_fg_hx_mul (
                .i_clk     (i_clk),
                .i_rst_n   (i_rst_n),
                .i_input_0 (fg[lane]),
                .i_input_1 (i_hx_value[lane*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]),
                .o_out     (fg_hx[lane])
            );

            mul_ugemm_sr_bipolar #(
                .WIDTH (SR_WIDTH)
            ) u_fg_ng_mul (
                .i_clk     (i_clk),
                .i_rst_n   (i_rst_n),
                .i_input_0 (fg[lane]),
                .i_input_1 (ng[lane]),
                .o_out     (fg_ng[lane])
            );

            // ng + (1 - fg_ng) + fg_hx: three 0/1 addends, so add_any popcounts
            // them and 1 - fg_ng is the inverted spike.
            add_any_bipolar #(
                .SCALE (OUT_SCALE),
                .WIDTH (WIDTH),
                .ENTRY (OUT_ENTRY)
            ) u_hy_add (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_input ({fg_hx[lane], ~fg_ng[lane], ng[lane]}),
                .o_out   (o_out[lane])
            );
        end
    endgenerate
endmodule
`default_nettype wire
