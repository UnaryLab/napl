`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// relu_cnt -- counter-based bipolar rate-coded ReLU (stateful).
//
// RTL counterpart of napl.operation.relu_cnt (src/napl/operation/relu.py).
// One input spike per cycle; an up/down saturating counter `acc` tracks the
// running bipolar value and clamps the output toward bipolar 0.
//
//   below_half = (acc < HALF)          // HALF = 2^(WIDTH-1)
//   o_out      = i_in | below_half     // force 1 unless input 0 and acc>=HALF
//   acc       <= clamp(acc + (o_out ? +1 : -1), 0, MAX)   // MAX = 2^WIDTH - 1
//
// Output is combinational in i_in and the current acc (pp_delay = 0); acc
// advances on each posedge i_clk. Active-low i_rst_n reloads acc = HALF, the
// model's reset() state.
//==============================================================================
module relu_cnt #(
    parameter WIDTH = 3
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,    // input spike (bipolar rate-coded)
    output wire o_out    // ReLU output spike
);
    localparam [WIDTH-1:0] MAX  = {WIDTH{1'b1}};        // 2^WIDTH - 1
    localparam [WIDTH-1:0] HALF = (1 << (WIDTH - 1));   // 2^(WIDTH-1)

    reg [WIDTH-1:0] acc;

    wire below_half = (acc < HALF);

    assign o_out = i_in | below_half;

    // Up/down saturating counter: +1 when o_out, -1 otherwise, clamped to [0, MAX].
    wire [WIDTH-1:0] acc_next =
        o_out ? ((acc == MAX) ? MAX : acc + 1'b1)
              : ((acc == {WIDTH{1'b0}}) ? {WIDTH{1'b0}} : acc - 1'b1);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= HALF;
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
