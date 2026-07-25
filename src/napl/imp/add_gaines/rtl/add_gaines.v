`timescale 1ns/1ps
`default_nettype none

// Gaines adder. SCALED selects the RNG-driven MUX or the unscaled OR circuit
// at elaboration. Verify from src/napl/imp with: make test OP=add_gaines
module add_gaines #(
    parameter integer SCALED = 1,
    parameter integer ENTRY = 8,
    parameter integer SELECT_WIDTH = 3
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire             i_clk,
    input  wire             i_rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [ENTRY-1:0] i_in,
    output wire             o_out
);
    generate
        if (SCALED != 0) begin : g_scaled
            reg [SELECT_WIDTH-1:0] select_index;
            reg [SELECT_WIDTH-1:0] select_rom [0:ENTRY-1];

            initial $readmemb("vec/add_gaines_rom.hex", select_rom);

            assign o_out = i_in[select_rom[select_index]];

            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    select_index <= {SELECT_WIDTH{1'b0}};
                else
                    select_index <= select_index
                        + {{(SELECT_WIDTH-1){1'b0}}, 1'b1};
            end
        end else begin : g_unscaled
            assign o_out = |i_in;
        end
    endgenerate
endmodule

`default_nettype wire
