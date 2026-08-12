`timescale 1ns/1ps
`default_nettype none

// Unipolar Gaines divider with a saturating feedback counter and RNG ROM.
// State uses an asynchronous active-low reset to the Python reset values.
// Verify from src/napl/imp with: make test OP=div_gaines

module div_gaines_unipolar #(
    parameter integer WIDTH = 5  // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_dividend,
    input  wire i_divisor,
    output wire o_output
);
    localparam [WIDTH-1:0] CNT_MAX = {WIDTH{1'b1}};
    localparam [WIDTH-1:0] CNT_INIT = {
        1'b1, {(WIDTH-1){1'b0}}
    };

    reg [WIDTH-1:0] cnt;
    reg [WIDTH-1:0] rng_idx;
    reg [WIDTH-1:0] rng_rom [0:(1<<WIDTH)-1];
    wire [WIDTH-1:0] rng_value;
    wire increment;
    wire decrement;

    initial $readmemb("vec/div_gaines_rom.hex", rng_rom);

    assign rng_value = rng_rom[rng_idx];
    assign o_output = (cnt > rng_value);
    assign increment = i_dividend;
    assign decrement = o_output & i_divisor;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= CNT_INIT;
            rng_idx <= {WIDTH{1'b0}};
        end else begin
            rng_idx <= rng_idx + {{(WIDTH-1){1'b0}}, 1'b1};
            if (increment && !decrement && cnt < CNT_MAX)
                cnt <= cnt + {{(WIDTH-1){1'b0}}, 1'b1};
            else if (!increment && decrement && cnt > {WIDTH{1'b0}})
                cnt <= cnt - {{(WIDTH-1){1'b0}}, 1'b1};
        end
    end
endmodule

`default_nettype wire
