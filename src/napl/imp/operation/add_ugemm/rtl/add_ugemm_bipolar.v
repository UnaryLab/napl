`timescale 1ns/1ps
`default_nettype none

// Bipolar uGEMM adder. SCALED mirrors the Python configuration at elaboration.
// Verify from src/napl/imp with: make test OP=add_ugemm

module add_ugemm_bipolar #(
    parameter integer SCALED = 1,      // inherited from config['scaled']; tb overrides via `GEN_SCALED
    parameter integer ENTRY = 8,       // # addends (reduction dim); tb overrides via `GEN_ENTRY
    parameter integer COUNT_WIDTH = 4, // derived from ENTRY; tb overrides via `GEN_COUNT_WIDTH
    parameter integer ACC_WIDTH = 14   // accumulator width; tb overrides via `GEN_ACC_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire [ENTRY-1:0] i_input,
    output wire             o_output
);
    wire [COUNT_WIDTH-1:0] partial_count [0:ENTRY];
    assign partial_count[0] = {COUNT_WIDTH{1'b0}};

    genvar lane;
    generate
        for (lane = 0; lane < ENTRY; lane = lane + 1) begin : g_count
            assign partial_count[lane+1] = partial_count[lane]
                + {{(COUNT_WIDTH-1){1'b0}}, i_input[lane]};
        end
    endgenerate

    wire [COUNT_WIDTH-1:0] input_count = partial_count[ENTRY];

    generate
        if (SCALED != 0) begin : g_scaled
            reg [COUNT_WIDTH-1:0] accumulator;
            wire [COUNT_WIDTH:0] sum =
                {1'b0, accumulator} + {1'b0, input_count};
            wire [COUNT_WIDTH:0] bound = ENTRY[COUNT_WIDTH:0];
            wire fire = (sum >= bound);
            wire [COUNT_WIDTH-1:0] next_accumulator =
                fire
                ? (sum[COUNT_WIDTH-1:0] - bound[COUNT_WIDTH-1:0])
                : sum[COUNT_WIDTH-1:0];

            assign o_output = fire;

            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    accumulator <= {COUNT_WIDTH{1'b0}};
                else
                    accumulator <= next_accumulator;
            end
        end else begin : g_unscaled
            localparam integer OFFSET_X2 = ENTRY - 1;

            reg signed [ACC_WIDTH-1:0] accumulator_x2;
            reg signed [ACC_WIDTH-1:0] output_count_x2;
            wire signed [ACC_WIDTH-1:0] input_x2 =
                $signed({
                    1'b0,
                    {(ACC_WIDTH-COUNT_WIDTH-1){1'b0}},
                    input_count
                }) <<< 1;
            wire signed [ACC_WIDTH-1:0] offset_x2 =
                OFFSET_X2[ACC_WIDTH-1:0];
            wire signed [ACC_WIDTH-1:0] sum =
                accumulator_x2 + input_x2 - offset_x2;
            wire fire = (sum > output_count_x2);
            wire signed [ACC_WIDTH-1:0] next_output_count_x2 =
                output_count_x2
                + (fire ? {{(ACC_WIDTH-2){1'b0}}, 2'b10}
                        : {ACC_WIDTH{1'b0}});

            assign o_output = fire;

            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    accumulator_x2 <= {ACC_WIDTH{1'b0}};
                    output_count_x2 <= {ACC_WIDTH{1'b0}};
                end else begin
                    accumulator_x2 <= sum;
                    output_count_x2 <= next_output_count_x2;
                end
            end
        end
    endgenerate
endmodule

`default_nettype wire
