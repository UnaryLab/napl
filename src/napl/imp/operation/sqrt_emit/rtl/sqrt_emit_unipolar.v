`timescale 1ns/1ps
`default_nettype none
// Unipolar sqrt_emit equivalent. The accumulator uses feedback from a depth-2
// alternating shift register gated by o_output.
// Output is combinational (pp_delay=0); state advances each posedge.
// Active-low reset clears emit/acc and loads sr[i]=i%2.


module sqrt_emit_unipolar (
    input  wire i_clk,    // one posedge == one Python forward() timestep
    input  wire i_rst_n,  // active-low; maps to Python reset()
    input  wire i_input,     // input spike stream
    output wire o_output  // square-root spike stream
);
    // nsadd accumulator: add_scale's clamp is [-4, 3], but reset is 0, the
    // addend is nonnegative, and a fire stores acc_add - 1 >= 0, so acc is
    // stored in [0, 2].
    reg        [1:0] acc;
    // emit feedback bit (1-bit), depth-2 scrambler shift register.
    reg              emit_out;
    reg        [1:0] sr;      // sr[0] = oldest (read out), sr[1] = newest

    // --- nsadd combinational datapath ---------------------------------------
    wire [1:0]       in_sum  = {1'b0, i_input} + {1'b0, emit_out};   // 0,1,2
    // No reachable state has acc = 2 with emit_out = 1, so acc_pre stays at or
    // below add_scale's upper rail 3 and needs no clamp; the lower rail -4 is
    // below acc_pre.
    wire [2:0]       acc_pre = {1'b0, acc} + {1'b0, in_sum};         // 0..3
    assign o_output = |acc_pre;
    // subtract o_output only where acc_pre>=1, so acc_next in [0,2].
    wire [1:0]       acc_next = acc_pre[1:0] - {1'b0, o_output};

    // --- shift register / emit ----------------------------------------------------
    wire scrambled = sr[0];

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            acc      <= 2'd0;
            emit_out <= 1'b0;
            sr       <= 2'b10;          // sr[0]=0, sr[1]=1  (reg[i]=i%2)
        end else begin
            acc      <= acc_next;
            emit_out <= scrambled & o_output;
            sr       <= {~o_output, sr[1]};  // emit oldest, push (1-output) at tail
        end
    end
endmodule
`default_nettype wire
