`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// add_any_unipolar -- unipolar any-scale accumulating adder (stateful).
//
// RTL counterpart of napl.operation.add_any (src/napl/operation/add.py),
// unipolar variant. Per timestep the model does:
//   acc += (partial - offset); clamp(acc, acc_min, acc_max);
//   out  = (acc >= scale);     acc -= scale*out
// where partial is the reduced input partial sum (the model's dim=None path).
// Unipolar leaves offset = 0 (the offset is only set in the bipolar branch).
//
// The datapath is kept at 2x scale (A = 2*acc) for parity with the bipolar
// module (whose (ENTRY-SCALE)/2 offset can be a half-integer and needs it); here
// 2*offset = 0:
//   A += 2*partial;            clamp(A, -2^WIDTH, 2^WIDTH-2);
//   o_out = (A >= 2*SCALE);    A -= 2*SCALE*o_out
//   2*offset = 0,  TWO_SCL = 2*SCALE,  A in [ACC_LO, ACC_HI] = [-2^WIDTH, 2^WIDTH-2].
//
// SCALE, WIDTH, ENTRY are Verilog parameters inherited from the Python model's
// config (scale/width and the reduction dim ENTRY): the testbench overrides them
// with `GEN_SCALE / `GEN_WIDTH / `GEN_ENTRY (emitted by gen/gen_add_any.py from
// the same config test_add_any.py uses), so the verified hardware always tracks
// the simulator. The defaults here are only a standalone-elaboration fallback.
// Every bus width and magic constant below is derived from these parameters.
//
// Output is combinational from the current accumulator state (same cycle as the
// input), so pp_delay = 0. One Python forward() timestep == one posedge i_clk;
// active-low i_rst_n reproduces reset() (accumulator = 0).
//==============================================================================
module add_any_unipolar #(
    parameter integer SCALE = 128,   // inherited from config['scale']; tb overrides via `GEN_SCALE
    parameter integer WIDTH = 20,    // inherited from config['width']; tb overrides via `GEN_WIDTH
    parameter integer ENTRY = 128    // # addends (reduction dim);   tb overrides via `GEN_ENTRY
) (
    input  wire                  i_clk,
    input  wire                  i_rst_n,
    input  wire [IN_W-1:0]       i_in,     // per-timestep partial sum, range [0, ENTRY]
    output wire                  o_out     // unipolar rate-coded output spike
);
    // ---- ceil(log2(x)) constant function (Verilog-2001) ----
    function integer clog2;
        input integer value;
        integer v;
        begin
            v = value - 1;
            for (clog2 = 0; v > 0; clog2 = clog2 + 1)
                v = v >> 1;
        end
    endfunction

    // ---- derived sizing (all magic constants trace to the parameters) ----
    localparam integer TWO_OFS = 0;                  // 2*offset = 0 (unipolar)
    localparam integer TWO_SCL = 2 * SCALE;          // 2*scale
    localparam integer ACC_HI  = (2 ** WIDTH) - 2;   // 2*(2^(WIDTH-1)-1)
    localparam integer ACC_LO  = -(2 ** WIDTH);      // 2*(-2^(WIDTH-1))

    // input port holds partial in [0, ENTRY] (unsigned).
    localparam integer IN_W  = clog2(ENTRY + 1);
    // A = 2*acc; signed reg of width WIDTH+1 covers [-2^WIDTH, 2^WIDTH-1] ⊇ state.
    localparam integer ACC_W = WIDTH + 1;
    // pre-clamp sum |value| <= 2^WIDTH + 2*ENTRY + |TWO_OFS| <= 2^WIDTH + 3*ENTRY;
    // size a signed bus to hold it (clog2 magnitude + sign).
    localparam integer SUM_W = clog2((2 ** WIDTH) + 3 * ENTRY + 1) + 1;

    reg signed [ACC_W-1:0] acc;

    // Signed constants sized to the datapath so every compare/add is signed-vs-signed
    // (slice the 32-bit integer localparam down to SUM_W, sign preserved).
    wire signed [SUM_W-1:0] s_hi  = ACC_HI[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_lo  = ACC_LO[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_scl = TWO_SCL[SUM_W-1:0];
    wire signed [SUM_W-1:0] s_ofs = TWO_OFS[SUM_W-1:0];

    // ---- combinational: this cycle's output and next-cycle accumulator state ----
    // 2*partial as a signed value (i_in in [0,ENTRY]): zero-extend to SUM_W, then <<1.
    wire signed [SUM_W-1:0] in_ext = $signed({{(SUM_W-IN_W){1'b0}}, i_in});
    wire signed [SUM_W-1:0] two_p  = in_ext <<< 1;
    wire signed [SUM_W-1:0] sum   = $signed(acc) + two_p - s_ofs;   // + 2*partial - 2*offset
    wire signed [SUM_W-1:0] clmp  = (sum > s_hi) ? s_hi :
                                    (sum < s_lo) ? s_lo : sum;
    wire                    fire  = (clmp >= s_scl);
    // clmp is in [ACC_LO,ACC_HI] so it fits ACC_W signed; nxt = fired? clmp-TWO_SCL : clmp
    // stays within [ACC_LO, ACC_HI], also ACC_W signed.
    wire signed [ACC_W-1:0] nxt   = fire ? (clmp[ACC_W-1:0] - s_scl[ACC_W-1:0])
                                         : clmp[ACC_W-1:0];

    assign o_out = fire;

    // ---- sequential: advance the accumulator; i_rst_n low maps to reset() (acc=0) ----
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            acc <= {ACC_W{1'b0}};
        else
            acc <= nxt;
    end
endmodule
`default_nettype wire
