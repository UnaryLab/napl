`timescale 1ns/1ps
`default_nettype none
// Spike counter equivalent to napl.sim.operation.decode; pp_delay=0.
// The count is polarity independent: unipolar and bipolar differ only in how the
// count is scaled when read, so one bare module covers both.
// Generated WIDTH mirrors Python self.width = ceil(log2(config['timestep'])).


module decode #(
    parameter integer WIDTH = 4  // inherited from the model width; tb overrides via `GEN_WIDTH
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_spike,        // input spike stream
    output wire [WIDTH:0]   o_spike_count   // running count, including this cycle's spike
);
    // WIDTH+1 bits: the count reaches 2**WIDTH after a full timestep run.
    reg [WIDTH:0] count;

    assign o_spike_count = count + {{WIDTH{1'b0}}, i_spike};

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            count <= {(WIDTH+1){1'b0}};
        else
            count <= o_spike_count;
    end
endmodule
`default_nettype wire
