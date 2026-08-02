`timescale 1ns/1ps
`default_nettype none
// Fixed-width signabs helper matching Python. The current input updates acc in
// [0,7] before sign=(acc<4) and abs=sign^input are read.
// Active-low reset loads acc=4.


module div_iscb_signabs (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,
    output wire o_sign,
    output wire o_abs
);
    localparam [3:0] ACC_MAX = 4'd7;
    localparam [3:0] ACC_MED = 4'd4;

    reg [3:0] acc_q;

    // acc updated this timestep BEFORE sign/abs are read (per the Python model).
    wire        at_max   = (acc_q == ACC_MAX);
    wire        at_min   = (acc_q == 4'd0);
    wire [3:0]  acc_next =
        i_input ? (at_max ? ACC_MAX : (acc_q + 4'd1))
             : (at_min ? 4'd0    : (acc_q - 4'd1));

    wire sign = (acc_next < ACC_MED);
    assign o_sign = sign;
    assign o_abs  = sign ^ i_input;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc_q <= ACC_MED;
        else
            acc_q <= acc_next;
    end
endmodule
`default_nettype wire
