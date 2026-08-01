`timescale 1ns/1ps
`default_nettype none
// Unipolar correlated divider equivalent to napl.sim.operation.div_cordiv.
// The quotient selects the dividend on divisor spikes and otherwise reads the
// buffered quotient at the bit-reversed Gray-code Sobol index.
// Output is combinational (pp_delay=0); each posedge shifts on a divisor spike
// and advances the index. Generated DEPTH/WIDTH mirror Python.
// Active-low reset clears the buffer and index.
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
    reg [DEPTH-1:0] buffer_q;   // buffer_q[0] newest, buffer_q[DEPTH-1] oldest
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
        if (!i_rst_n) begin
            buffer_q[0] <= 1'b0;
            idx      <= {WIDTH{1'b0}};
        end else begin
            if (i_divisor)
                buffer_q[0] <= o_quotient;
            idx <= idx + 1'b1;
        end
    end

    genvar row;
    generate
        for (row = 1; row < DEPTH; row = row + 1) begin: g_buffer
            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n)
                    buffer_q[row] <= 1'b0;
                else if (i_divisor)
                    buffer_q[row] <= buffer_q[row-1];
            end
        end
    endgenerate
endmodule
`default_nettype wire
