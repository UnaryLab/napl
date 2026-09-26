`timescale 1ns/1ps
`default_nettype none

// Unipolar uGEMM adder. SCALED mirrors the Python configuration at elaboration.
// Verify from src/napl/imp with: make test OP=add_ugemm

module add_ugemm_unipolar #(
    parameter integer SCALED = 1,      // inherited from config['scaled']; tb overrides via `GEN_SCALED
    parameter integer ENTRY = 8,       // # addends (reduction dim); tb overrides via `GEN_ENTRY
    parameter integer COUNT_WIDTH = 4, // derived from ENTRY; tb overrides via `GEN_COUNT_WIDTH
    parameter integer ACC_WIDTH = 14   // accumulator width; tb overrides via `GEN_ACC_WIDTH_UNI
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
            // Padded by a full COUNT_WIDTH zeros and sliced so no replication is empty.
            /* verilator lint_off UNUSEDSIGNAL */
            wire [COUNT_WIDTH:0] lane_pad = {{COUNT_WIDTH{1'b0}}, i_input[lane]};
            /* verilator lint_on UNUSEDSIGNAL */
            assign partial_count[lane+1] = partial_count[lane] + lane_pad[COUNT_WIDTH-1:0];
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
            // gap_x2 is twice the Python accumulator minus twice the
            // emitted-spike count; the model fires when that gap is positive.
            reg signed [ACC_WIDTH-1:0] gap_x2;
            // ACC_WIDTH >= COUNT_WIDTH + 1 holds for every mapped size, so the
            // zero fill is at least one bit wide.
            wire signed [ACC_WIDTH-1:0] input_x2 =
                $signed({{(ACC_WIDTH-COUNT_WIDTH){1'b0}}, input_count}) <<< 1;
            wire signed [ACC_WIDTH-1:0] sum =
                gap_x2 + input_x2;
            wire fire = (sum > $signed({ACC_WIDTH{1'b0}}));

            assign o_output = fire;

            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    gap_x2 <= {ACC_WIDTH{1'b0}};
                else
                    gap_x2 <= sum
                        - (fire ? {{(ACC_WIDTH-2){1'b0}}, 2'b10}
                                : {ACC_WIDTH{1'b0}});
            end
        end
    endgenerate
endmodule

`default_nettype wire
