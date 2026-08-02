`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// relu_tc -- bipolar temporal-coded ReLU.
//
// The Python model increments cycle and accumulator before producing each
// output. The RTL exposes that next-state calculation combinationally, then
// commits both registers on the clock edge, so one vector row is one timestep.
// Once either value is above the threshold, only that predicate matters. Both
// registers therefore saturate at THRESHOLD+1 instead of wrapping during a
// continuous stream; this is equivalent to the Python model for all later
// timesteps and does not reset between encoded values.
//==============================================================================


module relu_tc #(
    parameter integer WIDTH = 8   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    localparam [WIDTH:0] THRESHOLD = (1 << (WIDTH - 1));
    localparam [WIDTH:0] PAST_THRESHOLD = THRESHOLD + 1'b1;

    reg [WIDTH:0] acc;
    reg [WIDTH:0] cycle;
    wire [WIDTH:0] acc_next;
    wire [WIDTH:0] cycle_next;

    assign acc_next = (acc >= PAST_THRESHOLD) ? PAST_THRESHOLD :
                      ((acc == THRESHOLD && i_input) ? PAST_THRESHOLD :
                      (acc + i_input));
    assign cycle_next = (cycle >= THRESHOLD) ? PAST_THRESHOLD :
                        (cycle + 1'b1);
    assign o_out = (cycle_next <= THRESHOLD) ?
                   (acc_next <= cycle_next) :
                   ((acc_next > THRESHOLD) & i_input);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            acc <= {WIDTH+1{1'b0}};
            cycle <= {WIDTH+1{1'b0}};
        end else begin
            acc <= acc_next;
            cycle <= cycle_next;
        end
    end
endmodule
`default_nettype wire
