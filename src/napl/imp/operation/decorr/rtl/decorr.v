`timescale 1ns/1ps
`default_nettype none
// decorr equivalent: one shuffle buffer of DEPTH positions per stream. A
// SEQ_LEN-entry ROM generated from the Python number sequences supplies both
// per-timestep positions; position DEPTH-1 passes the current input through, and
// every other position emits the bit stored there and stores the current input in
// its place. Both outputs are combinational (pp_delay=0). Active-low reset loads
// cell[s]=s%2 and clears the sequence counter.
// The circuit is identical for both polarities, so there is one bare module.


module decorr #(
    parameter integer DEPTH   = 4,    // inherited from config['depth']; tb overrides via `GEN_DEPTH
    parameter integer IDX_W   = 2,    // position width, ceil(log2(DEPTH)) with a floor of 1
    parameter integer SEQ_LEN = 256,  // inherited from config['timestep']
    parameter integer SEQ_W   = 8     // counter width, ceil(log2(SEQ_LEN)) with a floor of 1
) (
    input  wire i_clk,      // one posedge == one Python forward() timestep
    input  wire i_rst_n,    // active-low; maps to Python reset()
    input  wire i_input_0,  // current spike of the first stream
    input  wire i_input_1,  // current spike of the second stream
    output wire o_output_0, // reordered first stream
    output wire o_output_1  // reordered second stream
);
    localparam integer SEQ_LAST = SEQ_LEN - 1;
    localparam integer PASS_IDX = DEPTH - 1;   // the pass-through position

    // One ROM word per timestep holds both positions as {idx_1, idx_0}, written by
    // gen_decorr.py from the model's rand_seq_idx. The vvp cwd is
    // imp/operation/decorr/, so the path is relative to that directory.
    reg [2*IDX_W-1:0] idx_rom [0:SEQ_LEN-1];
    initial $readmemb("vec/decorr_rom.hex", idx_rom);

    reg  [SEQ_W-1:0]   seq_cnt;
    wire [2*IDX_W-1:0] idx_word = idx_rom[seq_cnt];
    wire [1:0]         in_bits = {i_input_1, i_input_0};
    wire [1:0]         out_bits;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n)
            seq_cnt <= {SEQ_W{1'b0}};
        else if (seq_cnt == SEQ_LAST[SEQ_W-1:0])
            seq_cnt <= {SEQ_W{1'b0}};
        else
            seq_cnt <= seq_cnt + 1'b1;
    end

    genvar stream;
    generate
        for (stream = 0; stream < 2; stream = stream + 1) begin : g_stream
            wire [IDX_W-1:0] idx = idx_word[stream*IDX_W +: IDX_W];

            // Position DEPTH-1 is the multiplexer's pass-through input and has no
            // storage cell behind it, so store_q[DEPTH-1] stays unused. DEPTH of 1
            // leaves the pass-through position alone.
            reg [DEPTH-1:0] store_q;
            integer slot;

            assign out_bits[stream] = (idx == PASS_IDX[IDX_W-1:0]) ? in_bits[stream]
                                                                  : store_q[idx];

            always @(posedge i_clk or negedge i_rst_n) begin
                if (!i_rst_n) begin
                    for (slot = 0; slot < DEPTH-1; slot = slot + 1)
                        store_q[slot] <= slot[0];
                end else if (idx != PASS_IDX[IDX_W-1:0]) begin
                    store_q[idx] <= in_bits[stream];
                end
            end
        end
    endgenerate

    assign o_output_0 = out_bits[0];
    assign o_output_1 = out_bits[1];
endmodule
`default_nettype wire
