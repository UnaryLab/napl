`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// signabs_interleave -- bipolar sign and magnitude with an interleaved counter.
//
// Outputs use the accumulator after the current input is folded in, matching
// signabs_interleave.forward(). The accumulator itself is committed at the
// following clock edge, so both output paths have zero cycle latency.
//==============================================================================
module signabs_interleave #(
    parameter integer WIDTH = 5   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_sign,
    output wire o_magnitude
);
    localparam [WIDTH-1:0] ACC_MAX = {WIDTH{1'b1}};
    localparam [WIDTH-1:0] ACC_HALF = (1 << (WIDTH - 1));

    reg [WIDTH-1:0] acc;
    reg [WIDTH-1:0] acc_next;

    always @(*) begin
        if (i_input) begin
            if (acc == ACC_MAX)
                acc_next = ACC_MAX;
            else
                acc_next = acc + 1'b1;
        end else begin
            if (acc == {WIDTH{1'b0}})
                acc_next = {WIDTH{1'b0}};
            else
                acc_next = acc - 1'b1;
        end
    end

    assign o_sign = (acc_next < ACC_HALF);
    assign o_magnitude = o_sign ^ acc_next[0];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= ACC_HALF;
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
