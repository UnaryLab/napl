`timescale 1ns/1ps
`default_nettype none

// Unipolar Gaines square root with squared-output counter feedback.
// State uses an asynchronous active-low reset to the Python reset values.
// Verify from src/napl/imp with: make test OP=sqrt_gaines

module sqrt_gaines_unipolar #(
    parameter integer WIDTH = 5  // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_output
);
    localparam [WIDTH-1:0] CNT_MAX = {WIDTH{1'b1}};
    // Only the top bit set.
    localparam [WIDTH-1:0] CNT_INIT = CNT_MAX ^ (CNT_MAX >> 1);
    localparam [WIDTH-1:0] ONE = 1;

    reg [WIDTH-1:0] cnt;
    reg [WIDTH-1:0] rng_idx;
    reg out_d;
    reg [WIDTH-1:0] rng_rom [0:(1<<WIDTH)-1];
    wire [WIDTH-1:0] rng_value;
    wire decrement;

    initial $readmemb("vec/sqrt_gaines_rom.hex", rng_rom);

    assign rng_value = rng_rom[rng_idx];
    assign o_output = (cnt > rng_value);
    assign decrement = o_output & out_d;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= CNT_INIT;
            rng_idx <= {WIDTH{1'b0}};
            out_d <= 1'b0;
        end else begin
            rng_idx <= rng_idx + ONE;
            out_d <= o_output;
            if (i_input && !decrement && cnt < CNT_MAX)
                cnt <= cnt + ONE;
            else if (!i_input && decrement && cnt > {WIDTH{1'b0}})
                cnt <= cnt - ONE;
        end
    end
endmodule

`default_nettype wire
