`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// div_cordiv -- correlated stochastic division, unipolar.
//
// RTL counterpart of napl.operation.div_cordiv (src/napl/operation/div.py).
// Unipolar only (the Python class is polarity_required=False, no polarity
// branch), so the module is the bare op name with no _unipolar/_bipolar postfix.
//
// Per cycle the model computes, from the current state:
//   rand_q   = buffer_q[rand_seq[idx]]            // buffered past quotient
//   quotient = i_divisor ? i_dividend : rand_q    // correlated select
// and then, only where i_divisor is a spike, shifts the new quotient into the
// circular buffer (buffer_q[r] <= buffer_q[r-1] for r>0; buffer_q[0] <= quotient);
// the cyclic index idx advances every cycle. o_quotient is combinational in the
// inputs and the current state (no input->output register), so pp_delay = 0.
//
// DEPTH is a Verilog parameter inherited from the Python model's config['depth']:
// the testbench overrides it with `GEN_DEPTH (emitted by gen/gen_div_cordiv.py
// from the same config test_div_cordiv.py uses), so the verified hardware always
// tracks the simulator. The default here is only a standalone-elaboration
// fallback. All bus widths (buffer_q rows, idx width) are derived from DEPTH.
//
// For the optimal/default DEPTH=2 the Sobol-derived index table is
// rand_seq = {0, 1}, i.e. rand_seq[idx] == idx, so rand_q == buffer_q[idx]; the
// validated config is DEPTH=2.
//
// Reset (active-low i_rst_n) reproduces the Python reset() state EXACTLY:
// buffer_q all-zero and idx = 0.
//==============================================================================
module div_cordiv #(
    parameter integer DEPTH = 2   // inherited from config['depth']; tb overrides via `GEN_DEPTH
) (
    input  wire i_clk,       // one posedge == one Python forward() timestep
    input  wire i_rst_n,     // active-low; maps to Python reset()
    input  wire i_dividend,  // dividend spike stream
    input  wire i_divisor,   // divisor spike stream (the correlation/select line)
    output wire o_quotient   // quotient spike stream
);
    // idx is ceil(log2(DEPTH)) bits wide; clog2 covers DEPTH>=2 (and DEPTH==1).
    localparam integer WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    reg [DEPTH-1:0] buffer_q;   // buffer_q[0] newest, buffer_q[DEPTH-1] oldest
    reg [WIDTH-1:0] idx;        // cyclic buffer-row index, advances each cycle

    // Combinational quotient: select the dividend where the divisor spikes,
    // else the buffered past quotient indexed by idx (== rand_seq[idx] at DEPTH=2).
    wire rand_q = buffer_q[idx];
    assign o_quotient = i_divisor ? i_dividend : rand_q;

    integer r;
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            buffer_q <= {DEPTH{1'b0}};
            idx      <= {WIDTH{1'b0}};
        end else begin
            // Shift the new quotient into the buffer only where divisor spikes.
            // Top-down so each row reads its un-updated source (matches the model).
            if (i_divisor) begin
                for (r = DEPTH-1; r > 0; r = r - 1)
                    buffer_q[r] <= buffer_q[r-1];
                buffer_q[0] <= o_quotient;
            end
            // idx = (idx + 1) % DEPTH. The Python model asserts DEPTH is a power
            // of two, so the WIDTH-bit counter wrap IS the modulo (idx==DEPTH-1
            // is all-ones, +1 wraps to 0).
            idx <= idx + 1'b1;
        end
    end
endmodule
`default_nettype wire
