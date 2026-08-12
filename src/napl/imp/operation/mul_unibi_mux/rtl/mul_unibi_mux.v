`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// mul_unibi_mux -- unipolar x bipolar multiplier built from a multiplexer.
//
// RTL counterpart of napl.sim.operation.mul_unibi_mux
// (src/napl/sim/operation/mul_unibi_mux.py). i_input_u carries the unipolar
// magnitude, i_input_b the bipolar signed value, and o_output is bipolar:
// v_z = p_u * (2 * p_b - 1). The port polarities are fixed by the operation, so
// there is no polarity variant.
//
// The unipolar spike selects between the bipolar stream and a free-running
// rate-0.5 toggle stream, giving p_z = p_u * p_b + (1 - p_u) / 2:
//
//   o_output = i_input_u ? i_input_b : q
//   q'       = ~q                          // toggles every timestep
//
// The toggle is the deterministic 0, 1, 0, 1, ... sequence of the model, so an
// i_input_u locked to the same parity biases the second leg away from rate 0.5;
// that bias is inherent to the design.
//
// The output is combinational over the current inputs and the held state, so
// pp_delay is 0; each posedge i_clk advances one forward() timestep.
// i_rst_n (active-low) maps to the Python reset(): q <- 0.
//
// Verify from src/napl/imp/: conda run -n napl make test OP=mul_unibi_mux
//==============================================================================


module mul_unibi_mux (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_u,   // unipolar spike stream, the multiplexer select
    input  wire i_input_b,   // bipolar spike stream
    output wire o_output     // bipolar product spike
);
    reg q;

    assign o_output = i_input_u ? i_input_b : q;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            q <= 1'b0;
        else
            q <= ~q;
    end
endmodule
`default_nettype wire
