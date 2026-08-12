`timescale 1ns/1ps
`default_nettype none
// signabs equivalent with a WIDTH-derived saturating accumulator.
// The current input produces acc_next before sign and magnitude are read, so
// outputs are combinational (pp_delay=0). Active-low reset loads ACC_MED.


module signabs #(
    parameter integer WIDTH = 3   // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // bipolar rate-coded input spike
    output wire o_sign,   // 1 == negative, 0 == positive (this timestep)
    output wire o_magnitude  // magnitude spike
);
    localparam integer ACC_MAX = (1 << WIDTH) - 1;       // saturating maximum
    localparam integer ACC_MED = (1 << (WIDTH - 1));     // midpoint / reset value

    // acc holds the value carried into this timestep; acc_next folds in i_input.
    reg  [WIDTH-1:0] acc;
    reg  [WIDTH-1:0] acc_next;

    always @(*) begin
        if (i_input) begin
            // +1, saturating at ACC_MAX.
            acc_next = (acc == ACC_MAX[WIDTH-1:0]) ? ACC_MAX[WIDTH-1:0]
                                                   : acc + 1'b1;
        end else begin
            // -1, saturating at 0.
            acc_next = (acc == {WIDTH{1'b0}}) ? {WIDTH{1'b0}}
                                              : acc - 1'b1;
        end
    end

    assign o_sign = (acc_next < ACC_MED[WIDTH-1:0]) ? 1'b1 : 1'b0;
    assign o_magnitude = o_sign ^ i_input;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= ACC_MED[WIDTH-1:0];
        else
            acc <= acc_next;
    end
endmodule
`default_nettype wire
