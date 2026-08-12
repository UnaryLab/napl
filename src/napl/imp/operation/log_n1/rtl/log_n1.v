`timescale 1ns/1ps
`default_nettype none

// Unipolar log(1 + x) series circuit with four input decorrelation delays.
// Coefficient streams come from the Python-generated ROM.
// State uses an asynchronous active-low reset to the Python reset values.
// The final stage multiplies the cascade by a decorrelated copy of x, the
// inverse-function counterpart of exp_n1's output NAND; the cascade and ROM
// structure are otherwise shared. Matching the sim model, the final multiply
// draws its x copy from the fourth delay tap in both coef[0] branches.
// Verify from src/napl/imp with: make test OP=log_n1

module log_n1 #(
    parameter integer WIDTH = 8  // inherited from ceil(log2(config['timestep'])); tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_output
);
    reg [WIDTH-1:0] coef_idx;
    reg input_d1;
    reg input_d2;
    reg input_d3;
    reg input_d4;
    reg [3:0] coef_rom [0:(1<<WIDTH)-1];
    wire [3:0] coef;
    wire n_1;
    wire n_2;
    wire n_3;
    wire n_4;

    initial $readmemb("vec/log_n1_rom.hex", coef_rom);

    assign coef = coef_rom[coef_idx];
    assign n_1 = ~(i_input & coef[3]);
    assign n_2 = ~(n_1 & input_d1 & coef[2]);
    assign n_3 = ~(n_2 & input_d2 & coef[1]);
    assign n_4 = ~(n_3 & input_d3 & coef[0]);
    assign o_output = coef[0] ? (n_4 & input_d4) : input_d4;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            coef_idx <= {WIDTH{1'b0}};
            input_d1 <= 1'b0;
            input_d2 <= 1'b0;
            input_d3 <= 1'b0;
            input_d4 <= 1'b0;
        end else begin
            coef_idx <= coef_idx + {{(WIDTH-1){1'b0}}, 1'b1};
            input_d4 <= input_d3;
            input_d3 <= input_d2;
            input_d2 <= input_d1;
            input_d1 <= i_input;
        end
    end
endmodule

`default_nettype wire
