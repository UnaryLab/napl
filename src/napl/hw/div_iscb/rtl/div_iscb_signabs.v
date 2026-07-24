`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// div_iscb_signabs -- sign/magnitude of a bipolar rate-coded spike (stateful).
//
// RTL counterpart of napl.sim.operation.signabs with width=3, as instantiated by
// div_iscb's bipolar_forward (src/napl/sim/operation/signabs.py). Internal helper
// for div_iscb_bipolar; not a standalone op.
//
//   acc  = clamp(acc + (2*in - 1), 0, 7)     (updated this cycle, then read)
//   sign = (acc < 4)        1 = negative, 0 = positive
//   abs  = sign ^ in
//
// Reset (active-low i_rst_n) loads the Python reset() state: acc = acc_med = 4.
//==============================================================================
module div_iscb_signabs (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_in,
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
        i_in ? (at_max ? ACC_MAX : (acc_q + 4'd1))
             : (at_min ? 4'd0    : (acc_q - 4'd1));

    wire sign = (acc_next < ACC_MED);
    assign o_sign = sign;
    assign o_abs  = sign ^ i_in;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc_q <= ACC_MED;
        else
            acc_q <= acc_next;
    end
endmodule
`default_nettype wire
