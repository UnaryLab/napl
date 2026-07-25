`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// div_iscb_unipolar -- in-stream correlation-based division, unipolar (stateful).
//
// RTL counterpart of napl.sim.operation.div_iscb with config polarity='unipolar'
// (src/napl/sim/operation/div_iscb.py), i.e. unipolar_forward: a skewed synchronizer
// (sync_skewed, width=3) feeding a correlated-division kernel (div_cordiv,
// depth=2, generator='sobol' -> rand_seq_idx = [0, 1]).
//
// One dividend spike and one divisor spike per cycle; the quotient spike is a
// purely combinational function of the current inputs and the CURRENT register
// state (the cnt counter, the depth-2 buffer, the read pointer), so the
// input->output latency is 0 (pp_delay=0). The registers update on the posedge
// for the next cycle.
//
// State / reset (active-low i_rst_n) reproduces the Python reset() EXACTLY:
//   sync_skewed.cnt  = 0   (range [0, 7], width 3)
//   div_cordiv.buf   = {0, 0}
//   div_cordiv.idx   = 0   (rand_seq_idx = [0, 1], so it reads buf[idx])
//==============================================================================
module div_iscb_unipolar (
    input  wire i_clk,        // one posedge == one Python forward() timestep
    input  wire i_rst_n,      // active-low; maps to Python reset()
    input  wire i_dividend,   // dividend spike stream
    input  wire i_divisor,    // divisor spike stream
    output wire o_quotient    // quotient spike stream
);
    // ----- sync_skewed (width = 3) ------------------------------------------
    localparam [2:0] CNT_MAX = 3'd7;

    reg [2:0] cnt_q;

    wire in_01_10   = (i_dividend ^ i_divisor);       // exactly one of the two is 1
    wire cnt_notmin = (cnt_q != 3'd0);
    wire cnt_notmax = (cnt_q != CNT_MAX);

    // select = cnt_notmin - (cnt_notmin + cnt_notmax) * dividend  (in {-1, 0, +1})
    // output_1 = dividend + in_01_10 * select, which is a single {0,1} spike:
    //   - if in_01_10 == 0: output_1 = dividend
    //   - if dividend == 0: output_1 = in_01_10 & cnt_notmin
    //   - if dividend == 1: output_1 = 1 - (in_01_10 & cnt_notmax)
    wire sync_out =
        (~in_01_10) ? i_dividend
        : (i_dividend ? (~cnt_notmax) : cnt_notmin);

    // cnt += in_01_10 ? (dividend ? +1 : -1) : 0, saturating in [0, 7]
    wire cnt_inc = in_01_10 &  i_dividend & cnt_notmax;
    wire cnt_dec = in_01_10 & ~i_dividend & cnt_notmin;

    // divisor passes through sync_skewed unchanged.
    wire sync_divisor = i_divisor;

    // ----- div_cordiv (depth = 2, rand_seq_idx = [0, 1]) --------------------
    reg       buf0_q, buf1_q;   // buffer rows
    reg       idx_q;            // read pointer (0 or 1)

    wire divisor_eq_1 = sync_divisor;                 // spike == 1
    wire rand_q       = idx_q ? buf1_q : buf0_q;      // buf[rand_seq_idx[idx]]
    wire quotient     = divisor_eq_1 ? sync_out : rand_q;

    assign o_quotient = quotient;

    // buffer update (top-down): buf[1] <= deq ? buf[0] : buf[1];
    //                           buf[0] <= deq ? quotient : buf[0];
    wire buf1_next = divisor_eq_1 ? buf0_q   : buf1_q;
    wire buf0_next = divisor_eq_1 ? quotient : buf0_q;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            cnt_q  <= 3'd0;
            buf0_q <= 1'b0;
            buf1_q <= 1'b0;
            idx_q  <= 1'b0;
        end else begin
            // saturating counter
            if (cnt_inc)
                cnt_q <= cnt_q + 3'd1;
            else if (cnt_dec)
                cnt_q <= cnt_q - 3'd1;

            // depth-2 buffer
            buf0_q <= buf0_next;
            buf1_q <= buf1_next;

            // read pointer advances every cycle: idx = (idx + 1) % 2
            idx_q <= ~idx_q;
        end
    end
endmodule
`default_nettype wire
