`timescale 1ns/1ps
`default_nettype none
// Online Sobol sequence generator: the Antonov-Saleev gray-code recurrence
// x_{n+1} = x_n ^ v[l(n)], where l(n) is the position of the least significant
// zero of the WIDTH-bit counter n and v is the direction-vector table of the
// Sobol dimension. o_rand holds x_n scaled to WIDTH bits, so the emitted values
// are the 2**WIDTH points k/2**WIDTH in Sobol order, restarting after one period.
// At the all-ones counter state the top one-hot position is taken, which XORs
// v[WIDTH-1] and returns the sequence to 0 with the counter wrap.
// The recurrence needs WIDTH at least 2, since its top one_hot term selects
// cnt[WIDTH-2:0], the negative part select cnt[-1:0] at WIDTH 1. Below that the
// whole period fits in a 2**WIDTH-entry table, so WIDTH 1 walks the two points
// 0 and 1 straight out of that table with the same timing.
// WIDTH is at least 1: a zero-bit sequence has no points.
// DIRVEC_FILE holds WIDTH lines of WIDTH binary bits, v[0] first, and is read
// relative to the simulation cwd. The parameter takes its width from the
// 24-character default, so an override is truncated to its last 24 characters:
// keep override paths at 24 characters or fewer.


module sobol #(
    parameter integer WIDTH = 8,  // sequence period is 2**WIDTH
    parameter DIRVEC_FILE = "vec/encode_dirvec_d1.hex"
) (
    input  wire             i_clk,
    input  wire             i_rst_n,
    input  wire             i_en,
    output reg  [WIDTH-1:0] o_rand
);
    generate
        if (WIDTH < 1) begin : g_bad_width
            // An unresolvable module reference makes iverilog fail the build when
            // the sequence would carry no bits.
            ERROR_sobol_WIDTH_must_be_at_least_1 u_bad ();
        end
    endgenerate

    genvar i;
    genvar j;

    generate
    if (WIDTH < 2) begin : g_table
        // One entry per point of the period, walked by a wrapping index: at
        // WIDTH 1 the two entries are 0 and 1.
        reg  [WIDTH-1:0] seq [0:2**WIDTH-1];
        reg  [WIDTH-1:0] idx;
        wire [WIDTH-1:0] idx_next = idx + 1'b1;

        initial begin
            seq[0]           = {WIDTH{1'b0}};
            seq[2**WIDTH-1]  = {WIDTH{1'b1}};
        end

        always @(posedge i_clk or negedge i_rst_n) begin
            if (!i_rst_n) begin
                idx    <= {WIDTH{1'b0}};
                o_rand <= {WIDTH{1'b0}};
            end else if (i_en) begin
                idx    <= idx_next;
                o_rand <= seq[idx_next];
            end
        end
    end else begin : g_recurrence
        reg [WIDTH-1:0] dir_vec [0:WIDTH-1];
        initial $readmemb(DIRVEC_FILE, dir_vec);

        reg  [WIDTH-1:0]       cnt;
        wire [WIDTH-1:0]       one_hot;
        wire [WIDTH*WIDTH-1:0] or_vec;
        wire [WIDTH-1:0]       vec;

        // one_hot[i] marks the least significant zero of cnt; the top position also
        // covers the all-ones state, which carries the same direction vector.
        assign one_hot[0] = ~cnt[0];

        for (i = 1; i < WIDTH-1; i = i + 1) begin : g_one_hot
            assign one_hot[i] = ~cnt[i] & (&cnt[i-1:0]);
        end

        assign one_hot[WIDTH-1] = &cnt[WIDTH-2:0];

        // Chained OR of the selected table entries: one_hot picks at most one, so
        // the chain is a mux over the direction-vector table.
        assign or_vec[0 +: WIDTH] = one_hot[0] ? dir_vec[0] : {WIDTH{1'b0}};

        for (j = 1; j < WIDTH; j = j + 1) begin : g_or_vec
            assign or_vec[j*WIDTH +: WIDTH] =
                or_vec[(j-1)*WIDTH +: WIDTH] | (one_hot[j] ? dir_vec[j] : {WIDTH{1'b0}});
        end

        assign vec = or_vec[(WIDTH-1)*WIDTH +: WIDTH];

        always @(posedge i_clk or negedge i_rst_n) begin
            if (!i_rst_n) begin
                cnt    <= {WIDTH{1'b0}};
                o_rand <= {WIDTH{1'b0}};
            end else if (i_en) begin
                cnt    <= cnt + {{(WIDTH-1){1'b0}}, 1'b1};
                o_rand <= o_rand ^ vec;
            end
        end
    end
    endgenerate
endmodule
`default_nettype wire
