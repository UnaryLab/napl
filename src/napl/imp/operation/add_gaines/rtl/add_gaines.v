`timescale 1ns/1ps
`default_nettype none

// Gaines adder. SCALED selects the RNG-driven MUX or the unscaled OR circuit
// at elaboration. Verify from src/napl/imp with: make test OP=add_gaines
//
// Scaled mode holds one ROM row per input and steps a free-running SELECT_WIDTH
// counter through it, so the ROM depth ENTRY and the counter span 2**SELECT_WIDTH
// are the same number. The Python model accepts a power-of-two entry only; the
// generate guard below is that restriction at elaboration.

module add_gaines #(
    parameter integer SCALED = 1,       // inherited from config['scaled']; tb overrides via `GEN_SCALED
    parameter integer ENTRY = 8,        // inherited from config['entry']; tb overrides via `GEN_ENTRY
    parameter integer SELECT_WIDTH = 3  // inherited from log2(config['entry']); tb overrides via `GEN_SELECT_WIDTH
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire             i_clk,
    input  wire             i_rst_n,
    /* verilator lint_on UNUSEDSIGNAL */
    input  wire [ENTRY-1:0] i_input,
    output wire             o_output
);
    generate
        if (SCALED != 0 && ENTRY != (1 << SELECT_WIDTH)) begin : g_bad_entry
            // An unresolvable module reference makes iverilog fail the build when
            // the selector span and the ROM depth differ.
            ERROR_add_gaines_ENTRY_must_equal_two_to_the_SELECT_WIDTH u_bad ();
        end
    endgenerate

    generate
        if (SCALED != 0) begin : g_scaled
            reg [SELECT_WIDTH-1:0] select_index;
            reg [SELECT_WIDTH-1:0] select_rom [0:ENTRY-1];

            initial $readmemb("vec/add_gaines_rom.hex", select_rom);

            assign o_output = i_input[select_rom[select_index]];

            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    select_index <= {SELECT_WIDTH{1'b0}};
                else
                    select_index <= select_index
                        + {{(SELECT_WIDTH-1){1'b0}}, 1'b1};
            end
        end else begin : g_unscaled
            assign o_output = |i_input;
        end
    endgenerate
endmodule

`default_nettype wire
