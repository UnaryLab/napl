`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear_gaines (unipolar) over LANES output features.
// Each lane is one output feature. The layer's per-timestep partial count is
// sum_f x_f * w_f plus the optional bias spike, so the lane is IN_FEATURES weight
// comparators feeding IN_FEATURES mul_gaines_unipolar cells, plus one encode cell
// for the bias.
// SCALED selects the Gaines adder at elaboration: the scaled arm selects one
// addend from the add_gaines selection sequence; the non-scaled arm OR-reduces
// the addends, matching add_gaines at SCALED = 0.
// Weight and bias arrive as held fixed-point codes, not spikes: this layer holds
// one threshold sequence per input feature and generates both streams itself
// (its Python internal_encode is 'private').
// One weight-threshold generator per input feature sits outside the lane
// generate and is shared by every lane; the comparators cannot be shared, because
// each of the LANES x IN_FEATURES weight codes needs its own compare against the
// feature threshold of this timestep.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears every threshold generator, selector index, and encoder
// index, to match reset().
// In scaled mode ENTRY must equal 2**SCALE_WIDTH, so the free-running selector
// addresses every addend and nothing else, and the select ROM holds one row per
// addend. That is the power-of-two addend count the Python add_gaines model
// accepts, and it is the same relation add_gaines carries between its ENTRY and
// SELECT_WIDTH. The generate guard below enforces it at elaboration.


module linear_gaines_unipolar #(
    parameter integer IN_FEATURES = 15,  // weight columns;        tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;       tb overrides via `GEN_LANES
    parameter integer SEQ_WIDTH   = 8,   // ceil(log2(timestep));  tb overrides via `GEN_SEQ_WIDTH
    parameter integer SCALE_WIDTH = 4,   // round(log2(ENTRY));    tb overrides via `GEN_SCALE_WIDTH
    parameter integer HAS_BIAS    = 1,   // 1 adds a bias addend, 0 drops it
    parameter integer SCALED      = 1    // 1 selects the scaled adder, 0 the OR adder
) (
    input  wire                                    i_clk,
    input  wire                                    i_rst_n,
    input  wire [IN_FEATURES-1:0]                  i_input,
    input  wire [LANES*IN_FEATURES*(SEQ_WIDTH+1)-1:0] i_weight,  // lane l, feature f at [(l*IN_FEATURES + f)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [LANES*(SEQ_WIDTH+1)-1:0]          i_bias,       // lane l at [l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                        o_output
);


    localparam integer OPW     = SEQ_WIDTH + 1;          // fixed-point operand width
    localparam integer ENTRY   = IN_FEATURES + HAS_BIAS; // addends per lane


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail when the scaled selector span does not match the addend count, and
    // when the scaled adder has fewer than two addends, which Python rejects; the
    // module name records only the span term.
    generate
        if (SCALED != 0 && (ENTRY < 2 || ENTRY != (1 << SCALE_WIDTH))) begin : g_bad_scale_width
            ERROR_linear_gaines_SCALE_WIDTH_must_match_ENTRY u_bad ();
        end
    endgenerate

    // Weight thresholds, one field per input feature, shared by every lane. The
    // model holds one Sobol dimension per feature, so each field is its own online
    // generator over that dimension's direction vectors. The table paths carry a
    // two-digit feature index, so IN_FEATURES is at most 100.
    wire [IN_FEATURES*SEQ_WIDTH-1:0] w_row;

    genvar seq;
    generate
        for (seq = 0; seq < IN_FEATURES; seq = seq + 1) begin : g_w_seq
            localparam [7:0] TENS = "0" + (seq / 10);
            localparam [7:0] ONES = "0" + (seq % 10);

            sobol #(
                .WIDTH       (SEQ_WIDTH),
                .DIRVEC_FILE ({"vec/lg_w", TENS, ONES, "_dv.hex"})
            ) u_w_seq (
                .i_clk   (i_clk),
                .i_rst_n (i_rst_n),
                .i_en    (1'b1),
                .o_rand  (w_row[seq*SEQ_WIDTH +: SEQ_WIDTH])
            );
        end
    endgenerate

    reg  [SCALE_WIDTH-1:0] select_rom [0:ENTRY-1];
    reg  [SCALE_WIDTH-1:0] select_idx;
    wire [SCALE_WIDTH-1:0] select;
    localparam [SCALE_WIDTH-1:0] SEL_ZERO = 0;
    localparam [SCALE_WIDTH-1:0] SEL_ONE  = 1;

    initial $readmemb("vec/gaines_rom.hex", select_rom);

    assign select = select_rom[select_idx];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            select_idx <= SEL_ZERO;
        else
            select_idx <= select_idx + SEL_ONE;
    end

    genvar lane, feature;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : g_lane
            wire [ENTRY-1:0] addend;

            for (feature = 0; feature < IN_FEATURES; feature = feature + 1) begin : g_mul
                wire w_spike;

                // The model's inlined encoder comparison: the held weight code
                // against this timestep's threshold for this input feature.
                assign w_spike = i_weight[(lane*IN_FEATURES + feature)*OPW +: OPW]
                               > {1'b0, w_row[feature*SEQ_WIDTH +: SEQ_WIDTH]};

                mul_gaines_unipolar u_mul (
                    .i_input_0 (i_input[feature]),
                    .i_input_1 (w_spike),
                    .o_output  (addend[feature])
                );
            end

            // The bias bit advances once per timestep on its own sequence, so it
            // is the plain encoder over that sequence.
            if (HAS_BIAS != 0) begin : g_bias
                encode #(
                    .WIDTH       (SEQ_WIDTH),
                    .FRAC        (SEQ_WIDTH),
                    .DIRVEC_FILE ("vec/lg_bias_dv.hex")
                ) u_bias (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_bias[lane*OPW +: OPW]),
                    .o_spike (addend[ENTRY-1])
                );
            end

            if (SCALED != 0) begin : g_scaled
                assign o_output[lane] = addend[select];
            end else begin : g_or
                // Non-scaled unipolar addition is count > 0, the OR reduction
                // add_gaines implements at SCALED = 0.
                add_gaines #(
                    .SCALED (0),
                    .ENTRY  (ENTRY)
                ) u_add (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (addend),
                    .o_output   (o_output[lane])
                );
            end
        end
    endgenerate
endmodule
`default_nettype wire
