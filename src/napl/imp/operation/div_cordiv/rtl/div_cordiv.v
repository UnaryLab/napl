`timescale 1ns/1ps
`default_nettype none
// Unipolar correlated divider equivalent to napl.sim.operation.div_cordiv.
// The quotient selects the dividend on divisor spikes and otherwise reads the
// buffered quotient at the bit-reversed Gray-code Sobol index.
// Output is combinational (pp_delay=0); each posedge shifts on a divisor spike
// and advances the index. Generated DEPTH/WIDTH mirror Python.
// Active-low reset clears the buffer and index.
// DEPTH must equal 2**WIDTH, so the WIDTH-bit index addresses every buffer row
// and no row is skipped; that also rules out an empty buffer. The generate guard
// below enforces it at elaboration and mapping.yaml carries the same restriction.


module div_cordiv #(
    parameter integer DEPTH = 2,  // inherited from config['depth']; tb overrides via `GEN_DEPTH
    parameter integer WIDTH = 1   // inherited from log2(config['depth']); tb overrides via `GEN_WIDTH
) (
    input  wire i_clk,       // one posedge == one Python forward() timestep
    input  wire i_rst_n,     // active-low; maps to Python reset()
    input  wire i_dividend,  // dividend spike stream
    input  wire i_divisor,   // divisor spike stream (the correlation/select line)
    output wire o_quotient   // quotient spike stream
);
    // Elaboration-time guard: an unresolvable module reference makes iverilog
    // fail the build when DEPTH is empty or is not the 2**WIDTH rows the index
    // spans. DEPTH 0 is called out on its own because 2**WIDTH reads 0 for a
    // negative WIDTH under integer arithmetic.
    generate
        if (DEPTH < 1 || (2 ** WIDTH) != DEPTH) begin : g_bad_depth
            ERROR_div_cordiv_DEPTH_must_equal_two_to_the_WIDTH u_bad ();
        end
    endgenerate

    reg [DEPTH-1:0] buffer_q;   // buffer_q[0] oldest, buffer_q[DEPTH-1] newest
    reg [WIDTH-1:0] idx;        // cyclic buffer-row index, advances each cycle

    // Combinational quotient: select the dividend where the divisor spikes,
    // else use the Python Sobol index.
    wire [WIDTH-1:0] rand_index;
    assign rand_index[0] = idx[WIDTH-1];
    genvar bit_index;
    generate
        for (bit_index = 0; bit_index < WIDTH-1; bit_index = bit_index + 1) begin : g_sobol
            assign rand_index[WIDTH-1-bit_index] =
                idx[bit_index] ^ idx[bit_index+1];
        end
    endgenerate
    wire rand_q = buffer_q[rand_index];
    assign o_quotient = i_divisor ? i_dividend : rand_q;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            idx <= {WIDTH{1'b0}};
        else
            idx <= idx + 1'b1;
    end

    genvar row;
    generate
        for (row = 0; row < DEPTH - 1; row = row + 1) begin: g_shift
            if ((row % 2) == 0) begin: g_even
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        buffer_q[row] <= 1'b0;
                    else if (i_divisor)
                        buffer_q[row] <= buffer_q[row + 1];
                end
            end else begin: g_odd
                always @(posedge i_clk or negedge i_rst_n) begin
                    if (!i_rst_n)
                        buffer_q[row] <= 1'b1;
                    else if (i_divisor)
                        buffer_q[row] <= buffer_q[row + 1];
                end
            end
        end
        if (((DEPTH - 1) % 2) == 0) begin: g_last_even
            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    buffer_q[DEPTH - 1] <= 1'b0;
                else if (i_divisor)
                    buffer_q[DEPTH - 1] <= o_quotient;
            end
        end else begin: g_last_odd
            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    buffer_q[DEPTH - 1] <= 1'b1;
                else if (i_divisor)
                    buffer_q[DEPTH - 1] <= o_quotient;
            end
        end
    endgenerate
endmodule
`default_nettype wire
