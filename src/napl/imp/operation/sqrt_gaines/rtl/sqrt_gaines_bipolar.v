`timescale 1ns/1ps
`default_nettype none

// Bipolar Gaines square root with XNOR squared-output counter feedback.
// State uses an asynchronous active-low reset to the Python reset values.
// Verify from src/napl/imp with: make test OP=sqrt_gaines
module sqrt_gaines_bipolar #(
    parameter integer WIDTH = 5  // inherited from config['width']; tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_out
);
    localparam [WIDTH-1:0] CNT_MAX = {WIDTH{1'b1}};
    localparam [WIDTH-1:0] CNT_INIT = {
        1'b1, {(WIDTH-1){1'b0}}
    };

    reg [WIDTH-1:0] cnt;
    reg [WIDTH-1:0] rng_idx;
    reg out_d;
    reg [WIDTH-1:0] rng_rom [0:(1<<WIDTH)-1];
    wire [WIDTH-1:0] rng_value;
    wire decrement;

    initial $readmemb("vec/sqrt_gaines_rom.hex", rng_rom);

    assign rng_value = rng_rom[rng_idx];
    assign o_out = (cnt > rng_value);
    assign decrement = ~(o_out ^ out_d);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt <= CNT_INIT;
            rng_idx <= {WIDTH{1'b0}};
            out_d <= 1'b0;
        end else begin
            rng_idx <= rng_idx + {{(WIDTH-1){1'b0}}, 1'b1};
            out_d <= o_out;
            if (i_input && !decrement && cnt < CNT_MAX)
                cnt <= cnt + {{(WIDTH-1){1'b0}}, 1'b1};
            else if (!i_input && decrement && cnt > {WIDTH{1'b0}})
                cnt <= cnt - {{(WIDTH-1){1'b0}}, 1'b1};
        end
    end
endmodule

`default_nettype wire
