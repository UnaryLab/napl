`timescale 1ns/1ps
`default_nettype none
// Exhaustive state probe for the coverage oracle in gen/gen_sqrt_traceiscb.py.
// Drives both RTL variants over all 4096 twelve-bit input words, each from its
// own reset, and prints the register state and the input bit standing before
// every posedge:
//
//     <uni trace,dff,buf0,buf1,idx> <bi trace,dff,buf0,buf1,idx> <bi acc>
//     <uni shuffle cells> <bi shuffle cells> <shuffle position counter> <in>
//
// The generator replays its Python step() against the printed trajectory cycle
// by cycle, so the recurrence it uses is pinned to the hardware rather than to
// its own arithmetic. `make test` compiles tb/sqrt_traceiscb_tb.v only; this
// file is built by the generator.


module sqrt_traceiscb_probe;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_uni;
    wire out_bi;

    integer word, k;

    sqrt_traceiscb_unipolar dut_uni (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input (in_bit),
        .o_output(out_uni)
    );

    sqrt_traceiscb_bipolar dut_bi (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input (in_bit),
        .o_output(out_bi)
    );

    initial begin
        clk    = 1'b0;
        rst_n  = 1'b1;
        in_bit = 1'b0;
        for (word = 0; word < 4096; word = word + 1) begin
            rst_n = 1'b0;
            #1;
            rst_n = 1'b1;
            #1;
            for (k = 11; k >= 0; k = k - 1) begin
                in_bit = (word >> k) & 1'b1;
                #1;
                // Position DEPTH-1 of a shuffle buffer is the pass-through path
                // and has no storage cell, so only cells 0 to DEPTH-2 print.
                $display("%b%b%b%b%b %b%b%b%b%b %b %b %b %0d %b",
                         dut_uni.trace_q, dut_uni.dff_q, dut_uni.buf0_q,
                         dut_uni.buf1_q, dut_uni.idx_q,
                         dut_bi.trace_q, dut_bi.dff_q, dut_bi.buf0_q,
                         dut_bi.buf1_q, dut_bi.idx_q, dut_bi.acc_q,
                         dut_uni.u_decorr.g_stream[0].store_q[2:0],
                         dut_bi.u_decorr.g_stream[0].store_q[2:0],
                         dut_uni.u_decorr.seq_cnt, in_bit);
                clk = 1'b1;
                #1;
                clk = 1'b0;
                #1;
            end
        end
        $finish;
    end
endmodule
`default_nettype wire
