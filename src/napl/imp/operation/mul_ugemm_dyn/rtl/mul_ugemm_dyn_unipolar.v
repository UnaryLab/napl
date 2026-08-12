`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_ugemm_dyn_unipolar -- shift-register decorrelated unary multiplication.
//
// This is the scalar RTL counterpart of napl.sim.operation.mul_ugemm_dyn with
// polarity="unipolar". The current count and RNG threshold produce the output
// before the input streams update the register, matching one Python timestep.
// The count is initialized lazily on the first forward() call in Python, so the
// first-call flag selects the alternating-register sum while count remains at
// its exact post-reset value of zero.
//
// WIDTH is inherited from config['width']; DEPTH and every internal size derive
// from it. The generated RNG ROM comes from the same Python model config.
//==============================================================================


module mul_ugemm_dyn_unipolar #(
    parameter integer WIDTH = 4   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);
    localparam integer DEPTH = (1 << WIDTH);

    reg [DEPTH-1:0] reg_q;
    reg [WIDTH:0] count;
    reg [WIDTH-1:0] head;
    reg [WIDTH-1:0] rng_idx;
    reg first_call;
    reg [WIDTH-1:0] rng_rom [0:DEPTH-1];

    wire [WIDTH:0] count_base;
    wire [WIDTH:0] removed;
    wire threshold_hit;
    wire [WIDTH:0] count_next;

    initial $readmemb("vec/mul_ugemm_dyn_rom.hex", rng_rom);

    assign count_base = first_call ? (DEPTH / 2) : count;
    assign removed = {{WIDTH{1'b0}}, reg_q[head]};
    assign threshold_hit = (count_base > {1'b0, rng_rom[rng_idx]});
    assign o_output = i_input_0 & threshold_hit;

    // count_base is the sum represented by reg_q before this cycle's write.
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
            first_call <= 1'b1;
        end else begin
            reg_q[head] <= i_input_1;
            count <= count_next;
            head <= (head == DEPTH - 1) ? {WIDTH{1'b0}} : head + 1'b1;
            rng_idx <= rng_idx + i_input_0;
            first_call <= 1'b0;
        end
    end
endmodule
`default_nettype wire
