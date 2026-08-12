`timescale 1ns/1ps
`default_nettype none
// napl.sim.module.linear_gaines (bipolar) over LANES output features.
// Each lane is one output feature. The layer's per-timestep partial count
// 2*sum(x*w) - sum(x) - sum(w) + IN_FEATURES is, term by term,
// sum_f (1 - x_f - w_f + 2*x_f*w_f) = sum_f XNOR(x_f, w_f), so the lane is
// IN_FEATURES weight comparators feeding IN_FEATURES mul_gaines_bipolar cells,
// plus one encode cell for the bias spike, which is one more addend.
// The Gaines adder selects one product or bias bit per timestep from the model's
// add_gaines selection sequence. Non-scaled bipolar addition is not supported.
// Weight and bias arrive as held fixed-point codes, not spikes: this layer holds
// one threshold sequence per input feature and generates both streams itself
// (Python internal_encode = True).
// The weight-threshold ROM is W_LEN x IN_FEATURES and sits outside the lane
// generate, shared by every lane; the comparators cannot be shared with it,
// because each of the LANES x IN_FEATURES weight codes needs its own compare
// against the feature threshold of this timestep.
// Output is combinational (pp_delay=0); each posedge advances one timestep.
// Active-low reset clears the threshold index and every encoder index, matching
// reset().
// ENTRY must equal 2**SCALE_WIDTH, so the free-running selector addresses every
// addend and nothing else, and the select ROM holds one row per addend. That is
// the power-of-two addend count the Python add_gaines model accepts, and it is
// the same relation add_gaines carries between its ENTRY and SELECT_WIDTH. The
// generate guard below enforces it at elaboration.


module linear_gaines_bipolar #(
    parameter integer IN_FEATURES = 15,  // weight columns;        tb overrides via `GEN_IN_FEATURES
    parameter integer LANES       = 8,   // output features;       tb overrides via `GEN_LANES
    parameter integer SEQ_WIDTH   = 8,   // ceil(log2(timestep));  tb overrides via `GEN_SEQ_WIDTH
    parameter integer SCALE_WIDTH = 4,   // round(log2(ENTRY));    tb overrides via `GEN_SCALE_WIDTH
    parameter integer HAS_BIAS    = 1    // 1 adds a bias addend, 0 drops it
) (
    input  wire                                       i_clk,
    input  wire                                       i_rst_n,
    input  wire [IN_FEATURES-1:0]                     i_input,
    input  wire [LANES*IN_FEATURES*(SEQ_WIDTH+1)-1:0] i_weight,  // lane l, feature f at [(l*IN_FEATURES + f)*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    input  wire [LANES*(SEQ_WIDTH+1)-1:0]             i_bias,    // lane l at [l*(SEQ_WIDTH+1) +: SEQ_WIDTH+1]
    output wire [LANES-1:0]                           o_output
);


    localparam integer OPW   = SEQ_WIDTH + 1;            // fixed-point operand width
    localparam integer W_LEN = 1 << SEQ_WIDTH;           // weight-threshold sequence length
    localparam integer ENTRY = IN_FEATURES + HAS_BIAS;   // addends per lane


    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when the selector span does not match the addend count.
    generate
        if (ENTRY != (1 << SCALE_WIDTH)) begin : g_bad_scale_width
            ERROR_linear_gaines_SCALE_WIDTH_must_match_ENTRY u_bad ();
        end
    endgenerate

    // Weight thresholds, one row per timestep and one field per input feature,
    // generated from w_num_seq. The row is shared by every lane.
    reg [IN_FEATURES*SEQ_WIDTH-1:0] w_rom [0:W_LEN-1];
    initial $readmemb("vec/linear_gaines_w.hex", w_rom);

    reg  [SEQ_WIDTH-1:0]             seq_idx;
    wire [IN_FEATURES*SEQ_WIDTH-1:0] w_row;

    reg  [SCALE_WIDTH-1:0] select_rom [0:ENTRY-1];
    reg  [SCALE_WIDTH-1:0] select_idx;
    wire [SCALE_WIDTH-1:0] select;

    initial $readmemb("vec/gaines_rom.hex", select_rom);

    assign w_row = w_rom[seq_idx];
    assign select = select_rom[select_idx];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            seq_idx <= {SEQ_WIDTH{1'b0}};
        else
            seq_idx <= seq_idx + {{(SEQ_WIDTH-1){1'b0}}, 1'b1};
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            select_idx <= {SCALE_WIDTH{1'b0}};
        else
            select_idx <= select_idx + {{(SCALE_WIDTH-1){1'b0}}, 1'b1};
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

                mul_gaines_bipolar u_mul (
                    .i_input_0 (i_input[feature]),
                    .i_input_1 (w_spike),
                    .o_output  (addend[feature])
                );
            end

            // The bias bit advances once per timestep on its own sequence, so it
            // is the plain encoder over that sequence.
            if (HAS_BIAS != 0) begin : g_bias
                encode #(
                    .WIDTH (SEQ_WIDTH),
                    .FRAC  (SEQ_WIDTH)
                ) u_bias (
                    .i_clk   (i_clk),
                    .i_rst_n (i_rst_n),
                    .i_input (i_bias[lane*OPW +: OPW]),
                    .o_spike (addend[ENTRY-1])
                );
            end

            assign o_output[lane] = addend[select];
        end
    endgenerate
endmodule
`default_nettype wire
