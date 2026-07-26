`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// div_iscb_bipolar -- in-stream correlation-based division, bipolar (stateful).
//
// RTL counterpart of napl.sim.operation.div_iscb with config polarity='bipolar'
// (src/napl/sim/operation/div_iscb.py), i.e. bipolar_forward:
//
//   (sign_d, abs_d) = signabs(dividend)            width 3
//   (sign_s, abs_s) = signabs(divisor)             width 3
//   u_abs_d         = bi2uni(abs_d)                  width 2
//   u_abs_s         = bi2uni(abs_s)                  width 2
//   u_abs_q         = unipolar_core(u_abs_d, u_abs_s)  (sync_skewed + div_cordiv)
//   bi_abs_q        = uni2bi(u_abs_q)                width 3
//   out             = sign_d ^ sign_s ^ bi_abs_q
//
// The unipolar magnitude path is exactly div_iscb_unipolar, instantiated here.
// The quotient spike is a combinational function of the current inputs and the
// CURRENT register state of every sub-block (each sub-block updates its own
// accumulator/counter on the posedge but produces its output from the
// already-updated value within the same timestep, matching the Python model),
// so the input->output latency is 0 (pp_delay=0).
//
// Reset (active-low i_rst_n) reaches every sub-block, reproducing the Python
// reset() state of each (signabs.acc=4, bi2uni/uni2bi.acc=0, the unipolar
// core's cnt=0 / buffer={0,0} / idx=0).
//==============================================================================
module div_iscb_bipolar (
    input  wire i_clk,        // one posedge == one Python forward() timestep
    input  wire i_rst_n,      // active-low; maps to Python reset()
    input  wire i_dividend,   // dividend spike stream (bipolar)
    input  wire i_divisor,    // divisor spike stream (bipolar)
    output wire o_quotient    // quotient spike stream (bipolar)
);
    wire sign_d, abs_d;
    wire sign_s, abs_s;
    wire u_abs_d, u_abs_s;
    wire u_abs_q;
    wire bi_abs_q;

    div_iscb_signabs u_sad (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_dividend), .o_sign(sign_d), .o_abs(abs_d)
    );
    div_iscb_signabs u_sas (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_divisor), .o_sign(sign_s), .o_abs(abs_s)
    );

    div_iscb_bi2uni u_b2u_d (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(abs_d), .o_out(u_abs_d)
    );
    div_iscb_bi2uni u_b2u_s (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(abs_s), .o_out(u_abs_s)
    );

    // Unipolar magnitude division core (sync_skewed + div_cordiv).
    div_iscb_unipolar u_core (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_dividend(u_abs_d), .i_divisor(u_abs_s),
        .o_quotient(u_abs_q)
    );

    div_iscb_uni2bi u_u2b (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(u_abs_q), .o_out(bi_abs_q)
    );

    assign o_quotient = sign_d ^ sign_s ^ bi_abs_q;
endmodule
`default_nettype wire
