`timescale 1ns/1ps
`default_nettype none

// Bipolar Gaines divider with delayed-divisor feedback and an RNG ROM.
// State uses an asynchronous active-low reset to the Python reset values.
// Verify from src/napl/imp with: make test OP=div_gaines
module div_gaines_bipolar #(
    parameter integer DEPTH = 5  // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_dividend,
    input  wire i_divisor,
    output wire o_out
);
    localparam [DEPTH-1:0] CNT_MAX = {DEPTH{1'b1}};
    localparam [DEPTH-1:0] CNT_INIT = {
        1'b1, {(DEPTH-1){1'b0}}
    };

    reg [DEPTH-1:0] cnt;
    reg [DEPTH-1:0] rng_idx;
    reg divisor_d;
    reg [DEPTH-1:0] rng_rom [0:(1<<DEPTH)-1];
    wire [DEPTH-1:0] rng_value;
    wire increment;
    wire decrement;

    initial $readmemb("vec/div_gaines_rom.hex", rng_rom);

    assign rng_value = rng_rom[rng_idx];
    assign o_out = (cnt > rng_value);
    assign increment = ~(divisor_d ^ i_divisor ^ o_out);
    assign decrement = i_dividend ^ i_divisor;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= CNT_INIT;
            rng_idx <= {DEPTH{1'b0}};
            divisor_d <= 1'b0;
        end else begin
            rng_idx <= rng_idx + {{(DEPTH-1){1'b0}}, 1'b1};
            divisor_d <= i_divisor;
            if (increment && !decrement && cnt < CNT_MAX)
                cnt <= cnt + {{(DEPTH-1){1'b0}}, 1'b1};
            else if (!increment && decrement && cnt > {DEPTH{1'b0}})
                cnt <= cnt - {{(DEPTH-1){1'b0}}, 1'b1};
        end
    end
endmodule

`default_nettype wire
