`timescale 1ns/1ps
`default_nettype none

// Unipolar tanh series circuit with input and intermediate delay taps.
// Coefficient streams come from the Python-generated ROM.
// State uses an asynchronous active-low reset to the Python reset values.
// Verify from src/napl/imp with: make test OP=tanh_p1
module tanh_p1 #(
    parameter integer WIDTH = 8
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,
    output wire o_out
);
    reg [WIDTH-1:0] coef_idx;
    reg [7:0] input_delay;
    reg [2:0] n_1_delay;
    reg [3:0] coef_rom [0:(1<<WIDTH)-1];
    wire [3:0] coef;
    wire input_d4;
    wire input_d8;
    wire n_1;
    wire n_2;
    wire n_3;
    wire n_4;
    wire n_5;
    wire n_2_term;
    wire n_3_term;
    wire n_4_term;

    initial $readmemb("vec/tanh_p1_rom.hex", coef_rom);

    assign coef = coef_rom[coef_idx];
    assign input_d4 = input_delay[3];
    assign input_d8 = input_delay[7];
    assign n_1 = i_in & input_d4;
    assign n_2 = ~n_1;
    assign n_2_term = coef[3] ? n_2 : 1'b1;
    assign n_3 = ~(n_2_term & n_1_delay[0]);
    assign n_3_term = coef[2] ? n_3 : 1'b1;
    assign n_4 = ~(n_3_term & n_1_delay[1]);
    assign n_4_term = coef[1] ? n_4 : 1'b1;
    assign n_5 = ~(n_4_term & n_1_delay[2]);
    assign o_out = input_d8 & (coef[0] ? n_5 : 1'b1);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            coef_idx <= {WIDTH{1'b0}};
            input_delay <= 8'b00000000;
            n_1_delay <= 3'b000;
        end else begin
            coef_idx <= coef_idx + {{(WIDTH-1){1'b0}}, 1'b1};
            input_delay <= {input_delay[6:0], i_in};
            n_1_delay <= {n_1_delay[1:0], n_1};
        end
    end
endmodule

`default_nettype wire
