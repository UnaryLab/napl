`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_shiftreg_bipolar -- shift-register decorrelated unary multiplication.
//
// This is the scalar RTL counterpart of napl.sim.operation.mul_shiftreg with
// polarity="bipolar". The positive path uses the input stream and the inverse
// path uses its complement, with independent RNG indices. Both paths observe
// the current shift-register count before the register update.
//==============================================================================
module mul_shiftreg_bipolar #(
    parameter integer WIDTH = 4   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_out
);
    localparam integer DEPTH = (1 << WIDTH);

    reg [DEPTH-1:0] reg_q;
    reg [WIDTH:0] count;
    reg [WIDTH-1:0] head;
    reg [WIDTH-1:0] rng_idx;
    reg [WIDTH-1:0] rng_idx_inv;
    reg first_call;
    reg [WIDTH-1:0] rng_rom [0:DEPTH-1];

    wire [WIDTH:0] count_base;
    wire [WIDTH:0] removed;
    wire threshold_hit;
    wire threshold_hit_inv;
    wire path;
    wire path_inv;
    wire input_0_inv;
    wire [WIDTH:0] count_next;

    initial $readmemb("vec/mul_shiftreg_rom.hex", rng_rom);

    assign count_base = first_call ? (DEPTH / 2) : count;
    assign removed = {{WIDTH{1'b0}}, reg_q[head]};
    assign threshold_hit = (count_base > {1'b0, rng_rom[rng_idx]});
    assign threshold_hit_inv = (count_base > {1'b0, rng_rom[rng_idx_inv]});
    assign input_0_inv = i_input_0 ? 1'b0 : 1'b1;
    assign path = i_input_0 & threshold_hit;
    assign path_inv = input_0_inv & !threshold_hit_inv;
    assign o_out = path | path_inv;

    assign count_next = (i_input_1 && !removed) ? (count_base + 1'b1) :
                        (!i_input_1 && removed) ? (count_base - 1'b1) :
                        count_base;

    integer index;
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            for (index = 0; index < DEPTH; index = index + 1)
                reg_q[index] <= index % 2;
            count <= {WIDTH+1{1'b0}};
            head <= {WIDTH{1'b0}};
            rng_idx <= {WIDTH{1'b0}};
            rng_idx_inv <= {WIDTH{1'b0}};
            first_call <= 1'b1;
        end else begin
            reg_q[head] <= i_input_1;
            count <= count_next;
            head <= (head == DEPTH - 1) ? {WIDTH{1'b0}} : head + 1'b1;
            rng_idx <= rng_idx + i_input_0;
            rng_idx_inv <= rng_idx_inv + input_0_inv;
            first_call <= 1'b0;
        end
    end
endmodule
`default_nettype wire
