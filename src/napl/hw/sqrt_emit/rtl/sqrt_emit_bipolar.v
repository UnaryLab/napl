`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// sqrt_emit_bipolar -- bipolar opportunistic-bit-inserting square root.
//
// RTL counterpart of napl.sim.operation.sqrt_emit (forward(), bipolar branch) in
// src/napl/sim/operation/sqrt_emit.py. The nsadd accumulator path and the o_out compute
// are IDENTICAL to the unipolar variant (nsadd is fixed unipolar in __init__),
// only the emit feedback differs: bipolar scrambles by AND-ing with bi2uni(out)
// rather than out directly. Per timestep t (one posedge i_clk):
//
//   in_sum  = i_in + emit_out                 // emit_out is feedback from t-1
//   acc_add = clamp(acc + in_sum, -4, 3)      // nsadd: unipolar, scale=1, w=3
//   o_out   = (acc_add >= 1)
//   acc'    = acc_add - o_out
//
//   scrambled = sr[0]                          // depth-2 shiftreg, reads oldest
//   sr'       = { ~o_out, sr[1] }              // push (1-output) at the tail
//   // bi2uni (width 2): acc_b' = clamp(acc_b + 2*o_out - 1, -2, 1)
//   out_uni   = (acc_b_add >= 1)
//   acc_b'    = acc_b_add - out_uni
//   emit_out' = scrambled & out_uni            // bipolar emit
//
// o_out is combinational in i_in given the cycle-t registers, so the input->
// output latency is 0 (pp_delay = 0).
//
// Reset (active-low i_rst_n) maps to the Python reset():
//   emit_out = 0, acc = 0, bi2uni acc_b = 0, shiftreg sr[i] = i % 2.
//==============================================================================
module sqrt_emit_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_in,     // input spike stream
    output wire o_out     // square-root spike stream
);
    reg signed [3:0] acc;      // nsadd accumulator, clamped to [-4, 3]
    reg signed [2:0] acc_b;    // bi2uni accumulator, clamped to [-2, 1]
    reg              emit_out; // emit feedback bit
    reg        [1:0] sr;       // sr[0] = oldest (read out), sr[1] = newest

    // --- nsadd combinational datapath ---------------------------------------
    wire [1:0]        in_sum  = {1'b0, i_in} + {1'b0, emit_out};   // 0,1,2
    wire signed [4:0] acc_pre = $signed({acc[3], acc}) + $signed({3'b000, in_sum});
    wire signed [4:0] acc_add =
        (acc_pre > 5'sd3)  ? 5'sd3  :
        (acc_pre < -5'sd4) ? -5'sd4 : acc_pre;
    assign o_out = (acc_add >= 5'sd1) ? 1'b1 : 1'b0;
    // acc_add in [-4,3]; subtract o_out only where acc_add>=1 -> acc_next [-4,2].
    wire signed [3:0] acc_next = acc_add[3:0] - $signed({3'b000, o_out});

    // --- bi2uni: out_uni = bi2uni(o_out) ------------------------------------
    // acc_b + 2*o_out - 1, clamped to [-2, 1].
    wire signed [3:0] accb_pre = $signed({acc_b[2], acc_b})
                               + $signed({2'b00, o_out, 1'b0})   // +2*o_out
                               - 4'sd1;
    wire signed [3:0] accb_add =
        (accb_pre > 4'sd1)  ? 4'sd1  :
        (accb_pre < -4'sd2) ? -4'sd2 : accb_pre;
    wire              out_uni  = (accb_add >= 4'sd1) ? 1'b1 : 1'b0;
    // accb_add in [-2,1]; subtract out_uni only where accb_add>=1 -> [-2,1].
    wire signed [2:0] accb_next = accb_add[2:0] - $signed({2'b00, out_uni});

    // --- shiftreg / emit ----------------------------------------------------
    wire scrambled = sr[0];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            acc      <= 4'sd0;
            acc_b    <= 3'sd0;
            emit_out <= 1'b0;
            sr       <= 2'b10;          // sr[0]=0, sr[1]=1  (reg[i]=i%2)
        end else begin
            acc      <= acc_next;
            acc_b    <= accb_next;
            emit_out <= scrambled & out_uni;
            sr       <= {~o_out, sr[1]};  // emit oldest, push (1-output) at tail
        end
    end
endmodule
`default_nettype wire
