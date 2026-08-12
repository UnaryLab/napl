`timescale 1ns/1ps
`default_nettype none
// Unipolar napl.sim.operation.clamp: a saturating counter that clamps the running
// estimate acc/timestep to the fixed-point band [LO, HI] (grid units 2^-FRACWIDTH).
// Exact-integer form of the Python model's float path: with acc the running sum of
// 0/1 value contributions, t the 1-based timestep, and em the emitted-spike count,
// the model fires when target_count(t) - em >= 0.5 with
//   target_count = clamp(acc/t, lo, hi) * t.
// Scaling both sides by 2^(FRACWIDTH+1) clears every fraction, so the compare is
// integer LHS >= RHS. Output is combinational (pp_delay=0); each posedge advances
// one timestep. Active-low reset clears ts/acc/em to match reset() (all zero).
// TIMESTEP bounds the run length between resets and sizes the counters; every bus
// width and constant derives from the parameters, and the tb overrides them from
// vec/clamp_params.vh.


module clamp_unipolar #(
    parameter integer LO        =  64,   // config['lo'] * 2^FRACWIDTH (grid units)
    parameter integer HI        = 192,   // config['hi'] * 2^FRACWIDTH (grid units)
    parameter integer FRACWIDTH = 8,     // config['fracwidth']
    parameter integer TIMESTEP  = 256    // run-length bound (max timesteps per reset)
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input,        // unipolar rate-coded input spike
    output wire o_output        // unipolar rate-coded clamped output spike
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
    localparam integer TS_W  = clog2(TIMESTEP + 1);   // timestep/emitted counter width
    // Signed compare bus: |LHS|,|RHS| <= (2*TIMESTEP+1)*2^FRACWIDTH; +1 for sign,
    // +2 headroom so acc<<(FRACWIDTH+1) and 2*HI*t_cur never overflow.
    localparam integer CMP_W = TS_W + FRACWIDTH + 3;

    reg  [TS_W-1:0]        ts;    // completed timesteps
    reg  signed [TS_W:0]   acc;   // running sum of 0/1 steps, in [0, TIMESTEP]
    reg  [TS_W-1:0]        em;    // emitted spike count, in [0, TIMESTEP]

    // ---- combinational: this cycle's output and next-cycle state ----
    // Current 1-based timestep and the accumulator after this input.
    wire signed [CMP_W-1:0] t_cur = $signed({{(CMP_W-TS_W){1'b0}}, ts}) + $signed({{(CMP_W-1){1'b0}}, 1'b1});
    wire signed [CMP_W-1:0] step  = i_input ? $signed({{(CMP_W-1){1'b0}}, 1'b1})
                                            : {CMP_W{1'b0}};
    wire signed [CMP_W-1:0] a_cur = $signed({{(CMP_W-TS_W-1){acc[TS_W]}}, acc}) + step;

    wire signed [CMP_W-1:0] lo_c  = LO[CMP_W-1:0];
    wire signed [CMP_W-1:0] hi_c  = HI[CMP_W-1:0];

    // Band test acc/t vs [lo, hi], cleared of the division: acc*2^F vs {lo,hi}*t.
    wire signed [CMP_W-1:0] a_scaled = a_cur <<< FRACWIDTH;
    wire below = a_scaled < (lo_c * t_cur);
    wire above = a_scaled > (hi_c * t_cur);

    // LHS = 2^(F+1) * target_count for each clamp regime.
    wire signed [CMP_W-1:0] lhs =
        below ? ((lo_c * t_cur) <<< 1) :
        above ? ((hi_c * t_cur) <<< 1) :
                (a_cur <<< (FRACWIDTH + 1));
    // RHS = 2^(F+1) * (em + 0.5) = (2*em + 1) * 2^F.
    wire signed [CMP_W-1:0] em_ext = $signed({{(CMP_W-TS_W){1'b0}}, em});
    wire signed [CMP_W-1:0] rhs = (((em_ext <<< 1) + $signed({{(CMP_W-1){1'b0}}, 1'b1})) <<< FRACWIDTH);

    wire fire = (lhs >= rhs);
    assign o_output = fire;

    // ---- sequential: advance state; i_rst_n low maps to reset() (all zero) ----
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            ts  <= {TS_W{1'b0}};
            acc <= {(TS_W+1){1'b0}};
            em  <= {TS_W{1'b0}};
        end else begin
            ts  <= ts + 1'b1;
            acc <= a_cur[TS_W:0];
            em  <= em + {{(TS_W-1){1'b0}}, fire};
        end
    end
endmodule
`default_nettype wire
