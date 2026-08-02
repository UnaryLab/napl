`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// relu_shiftreg -- bipolar rate-coded ReLU using a shift-register estimate.
//
// The output is combinational from the current delayed count and input. Python
// resets count to zero but initializes it from the alternating register during
// the first forward() call; first_call reproduces that transition exactly.
//==============================================================================


module relu_shiftreg #(
    parameter integer DEPTH = 8   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);


    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction


    localparam integer HEAD_WIDTH = (DEPTH <= 1) ? 1 : clog2(DEPTH);
    localparam integer HALF_CEIL = (DEPTH + 1) / 2;

    reg [DEPTH-1:0] reg_q;
    reg [DEPTH:0] count;
    reg [DEPTH:0] count_delayed;
    reg [HEAD_WIDTH-1:0] head;
    reg first_call;

    wire [DEPTH:0] count_base;
    wire below_half;
    wire [DEPTH:0] count_next;
    wire [DEPTH:0] count_delayed_next;
    wire removed;

    assign count_base = first_call ? (DEPTH / 2) : count;
    assign below_half = (count_delayed < HALF_CEIL);
    assign o_out = first_call ? 1'b1 : (below_half | i_input);
    assign removed = reg_q[head];
    assign count_next = (o_out && !removed) ? (count_base + 1'b1) :
                        (!o_out && removed) ? (count_base - 1'b1) :
                        count_base;
    assign count_delayed_next = count_base;

    integer index;
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            for (index = 0; index < DEPTH; index = index + 1)
                reg_q[index] <= index % 2;
            count <= {DEPTH+1{1'b0}};
            count_delayed <= {DEPTH+1{1'b0}};
            head <= {HEAD_WIDTH{1'b0}};
            first_call <= 1'b1;
        end else begin
            reg_q[head] <= o_out;
            count <= count_next;
            count_delayed <= count_delayed_next;
            head <= (head == DEPTH - 1) ? {HEAD_WIDTH{1'b0}} : head + 1'b1;
            first_call <= 1'b0;
        end
    end
endmodule
`default_nettype wire
