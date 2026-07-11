`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sqrt_emit_unipolar -- unipolar opportunistic-bit-inserting square root.
//
// RTL counterpart of napl.operation.sqrt_emit (forward(), unipolar branch) in
// src/napl/operation/sqrt.py. Per timestep t (one posedge i_clk):
//
//   in_sum  = i_in + emit_out                 // emit_out is feedback from t-1
//   acc_add = clamp(acc + in_sum, -4, 3)      // nsadd: unipolar, scale=1, w=3
//   o_out   = (acc_add >= 1)                   // non-scaled-add output
//   acc'    = acc_add - o_out                  // subtract scale where output=1
//
//   scrambled = sr[0]                          // depth-2 shiftreg, reads oldest
//   sr'       = { ~o_out, sr[1] }              // push (1-output) at the tail
//   emit_out' = scrambled & o_out              // unipolar emit
//
// o_out is combinational in i_in given the cycle-t registers, so the input->
// output latency is 0 (pp_delay = 0). acc, emit_out and the shift register are
// the only clocked state.
//
// Reset (active-low i_rst_n) maps to the Python reset():
//   emit_out = 0, acc = 0, shiftreg sr[i] = i % 2  (sr[0]=0, sr[1]=1).
//==============================================================================
module sqrt_emit_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_in,     // input spike stream
    output wire o_out     // square-root spike stream
);
    // nsadd accumulator: signed, clamped to [-4, 3] (width-3 add_any).
    reg signed [3:0] acc;
    // emit feedback bit (1-bit), depth-2 scrambler shift register.
    reg              emit_out;
    reg        [1:0] sr;      // sr[0] = oldest (read out), sr[1] = newest

    // --- nsadd combinational datapath ---------------------------------------
    wire [1:0]       in_sum  = {1'b0, i_in} + {1'b0, emit_out};   // 0,1,2
    wire signed [4:0] acc_pre = $signed({acc[3], acc}) + $signed({3'b000, in_sum});
    // clamp to [-4, 3]
    wire signed [4:0] acc_add =
        (acc_pre > 5'sd3)  ? 5'sd3  :
        (acc_pre < -5'sd4) ? -5'sd4 : acc_pre;
    assign o_out = (acc_add >= 5'sd1) ? 1'b1 : 1'b0;
    // acc_add in [-4,3]; subtract o_out only where acc_add>=1, so acc_next in
    // [-4,2] -> fits signed 4-bit with no truncation.
    wire signed [3:0] acc_next = acc_add[3:0] - $signed({3'b000, o_out});

    // --- shiftreg / emit ----------------------------------------------------
    wire scrambled = sr[0];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            acc      <= 4'sd0;
            emit_out <= 1'b0;
            sr       <= 2'b10;          // sr[0]=0, sr[1]=1  (reg[i]=i%2)
        end else begin
            acc      <= acc_next;
            emit_out <= scrambled & o_out;
            sr       <= {~o_out, sr[1]};  // emit oldest, push (1-output) at tail
        end
    end
endmodule
`default_nettype wire
