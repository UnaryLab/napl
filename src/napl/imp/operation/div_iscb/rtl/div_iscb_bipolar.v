`timescale 1ns/1ps
`default_nettype none
// Bipolar div_iscb equivalent: signabs(3), bi2uni(3), the unipolar magnitude
// core, then uni2bi(3), with quotient sign applied by XOR. The three converters
// are the standalone operation modules, elaborated at the width the Python model
// bakes in.
// Output is combinational (pp_delay=0); state advances each posedge.
// Active-low reset loads signabs acc=4, converter accumulators=0, and core
// history buf0=0, buf1=1.


module div_iscb_bipolar (
    input  wire i_clk,        // one posedge == one Python forward() timestep
    input  wire i_rst_n,      // active-low; maps to Python reset()
    input  wire i_dividend,   // dividend spike stream (bipolar)
    input  wire i_divisor,    // divisor spike stream (bipolar)
    output wire o_output    // quotient spike stream (bipolar)
);
    // Fixed in the Python model, not a config-derived size.
    localparam integer HELPER_WIDTH = 3;

    wire sign_d, abs_d;
    wire sign_s, abs_s;
    wire u_abs_d, u_abs_s;
    wire u_abs_q;
    wire bi_abs_q;

    signabs #(.WIDTH(HELPER_WIDTH)) u_sad (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_dividend), .o_sign(sign_d), .o_magnitude(abs_d)
    );
    signabs #(.WIDTH(HELPER_WIDTH)) u_sas (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(i_divisor), .o_sign(sign_s), .o_magnitude(abs_s)
    );

    bi2uni #(.WIDTH(HELPER_WIDTH)) u_b2u_d (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(abs_d), .o_output(u_abs_d)
    );
    bi2uni #(.WIDTH(HELPER_WIDTH)) u_b2u_s (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(abs_s), .o_output(u_abs_s)
    );

    // Unipolar magnitude division core (sync_skewed + div_cordiv).
    div_iscb_unipolar u_core (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_dividend(u_abs_d), .i_divisor(u_abs_s),
        .o_output(u_abs_q)
    );

    uni2bi #(.WIDTH(HELPER_WIDTH)) u_u2b (
        .i_clk(i_clk), .i_rst_n(i_rst_n),
        .i_input(u_abs_q), .o_output(bi_abs_q)
    );

    assign o_output = sign_d ^ sign_s ^ bi_abs_q;
endmodule
`default_nettype wire
