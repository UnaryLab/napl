`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// signabs_shiftreg -- bipolar sign and magnitude using a shift-register count.
//
// Python resets count to zero, then replaces it with the alternating-register
// sum on the first forward() call. first_call keeps that internal reset state
// exact while count_base exposes the first externally visible state.
//==============================================================================
module signabs_shiftreg #(
    parameter integer DEPTH = 8   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_sign,
    output wire o_magnitude
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
    reg [HEAD_WIDTH-1:0] head;
    reg first_call;

    wire [DEPTH:0] count_base;
    wire [DEPTH:0] count_next;
    wire removed;

    assign count_base = first_call ? (DEPTH / 2) : count;
    assign o_sign = (count_base < HALF_CEIL);
    assign o_magnitude = o_sign ^ i_input;
    assign removed = reg_q[head];
    assign count_next = (i_input && !removed) ? (count_base + 1'b1) :
                        (!i_input && removed) ? (count_base - 1'b1) :
                        count_base;

    integer index;
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            for (index = 0; index < DEPTH; index = index + 1)
                reg_q[index] <= index % 2;
            count <= {DEPTH+1{1'b0}};
            head <= {HEAD_WIDTH{1'b0}};
            first_call <= 1'b1;
        end else begin
            reg_q[head] <= i_input;
            count <= count_next;
            head <= (head == DEPTH - 1) ? {HEAD_WIDTH{1'b0}} : head + 1'b1;
            first_call <= 1'b0;
        end
    end
endmodule
`default_nettype wire
