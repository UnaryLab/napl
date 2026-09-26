`timescale 1ns/1ps
`default_nettype none
// Bipolar sqrt_emit equivalent. The unipolar accumulator path uses feedback
// from a depth-2 alternating shift register gated by bi2uni(o_output).
// Output is combinational (pp_delay=0); state advances each posedge.
// Active-low reset clears emit/accumulators and loads sr[i]=i%2.


module sqrt_emit_bipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_output  // square-root spike stream
);
    // nsadd accumulator. An exhaustive search from reset shows acc in {0, 1}:
    // no reachable state has acc = 1 with emit_out = 1, so acc_pre stays at or
    // below 2, below add_scale's upper rail 3, and one bit holds acc.
    reg              acc;
    reg signed [1:0] acc_b;    // bi2uni accumulator, stored in [-2, 0]
    reg              emit_out; // emit feedback bit
    reg        [1:0] sr;       // sr[0] = oldest (read out), sr[1] = newest

    // --- nsadd combinational datapath ---------------------------------------
    wire [1:0]        in_sum  = {1'b0, i_input} + {1'b0, emit_out};   // 0,1,2
    wire [1:0]        acc_pre = {1'b0, acc} + in_sum;                 // 0..2
    assign o_output = |acc_pre;
    // subtract o_output only where acc_pre>=1, so acc_next in {0, 1}.
    wire [1:0]        acc_next = acc_pre - {1'b0, o_output};

    // --- bi2uni: out_uni = bi2uni(o_output) ------------------------------------
    // acc_b + 2*o_output - 1, clamped to [-2, 1].
    wire signed [3:0] accb_pre = $signed({{2{acc_b[1]}}, acc_b})
                               + $signed({2'b00, o_output, 1'b0})   // +2*o_output
                               - 4'sd1;
    wire signed [3:0] accb_add =
        (accb_pre > 4'sd1)  ? 4'sd1  :
        (accb_pre < -4'sd2) ? -4'sd2 : accb_pre;
    wire              out_uni  = (accb_add >= 4'sd1) ? 1'b1 : 1'b0;
    // accb_add in [-2,1]; subtract out_uni only where accb_add>=1 -> [-2,0].
    wire signed [1:0] accb_next = accb_add[1:0] - $signed({1'b0, out_uni});

    // --- shift register / emit ----------------------------------------------------
    wire scrambled = sr[0];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            acc      <= 1'b0;
            acc_b    <= 2'sd0;
            emit_out <= 1'b0;
            sr       <= 2'b10;          // sr[0]=0, sr[1]=1  (reg[i]=i%2)
        end else begin
            acc      <= acc_next[0];
            acc_b    <= accb_next;
            emit_out <= scrambled & out_uni;
            sr       <= {~o_output, sr[1]};  // emit oldest, push (1-output) at tail
        end
    end
endmodule
`default_nettype wire
