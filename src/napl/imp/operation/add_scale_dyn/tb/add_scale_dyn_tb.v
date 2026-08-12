`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "add_scale_dyn/vec/add_scale_dyn_params.vh"
// Golden rows are <rst> <scale> <unipolar input/output> <bipolar input/output>. Reset
// precedes the marked row; outputs are checked before the posedge updates state.
// The scale column is scanned as characters and its length checked against SCALE_W,
// because %b would zero-extend a short column silently and pass at the wrong width.
// Co-sim: make test OP=add_scale_dyn


module add_scale_dyn_tb;
    localparam integer MAX_CHARS = `GEN_SCALE_W + 1;

    reg                    i_clk;
    reg                    i_rst_n;
    reg [`GEN_ENTRY-1:0]   i_input_uni;
    reg [`GEN_ENTRY-1:0]   i_input_bi;
    reg [`GEN_SCALE_W-1:0] i_scale;
    wire                   o_uni;
    wire                   o_bi;

    add_scale_dyn_unipolar #(
        .SCALE_W(`GEN_SCALE_W),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY)
    ) dut_uni (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni),
        .i_scale   (i_scale),
        .o_output  (o_uni)
    );

    add_scale_dyn_bipolar #(
        .SCALE_W(`GEN_SCALE_W),
        .WIDTH(`GEN_WIDTH),
        .ENTRY(`GEN_ENTRY)
    ) dut_bi (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi),
        .i_scale   (i_scale),
        .o_output  (o_bi)
    );

    // 10ns clock
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_flag, exp_uni, exp_bi;
    reg [MAX_CHARS*8-1:0] tok_scale;


    // Characters $fscanf("%s") stored: the bit width the golden row carries.
    function integer token_len;
        input [MAX_CHARS*8-1:0] token;
        integer index;
        begin
            token_len = 0;
            for (index = 0; index < MAX_CHARS; index = index + 1)
                if (token[index*8 +: 8] != 8'h00)
                    token_len = token_len + 1;
        end
    endfunction


    // '0'/'1' characters to bits: character c from the right is bit c.
    function [MAX_CHARS-1:0] token_bits;
        input [MAX_CHARS*8-1:0] token;
        integer index;
        begin
            token_bits = {MAX_CHARS{1'b0}};
            for (index = 0; index < MAX_CHARS; index = index + 1)
                token_bits[index] = token[index*8];
        end
    endfunction


    initial begin
        fd = $fopen("vec/add_scale_dyn.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/add_scale_dyn.vec (run `make vectors` first)");
            $finish;
        end

        i_input_uni = {`GEN_ENTRY{1'b0}};
        i_input_bi = {`GEN_ENTRY{1'b0}};
        i_scale = {`GEN_SCALE_W{1'b0}};
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %s %b %b %b %b\n",
                rst_flag, tok_scale, i_input_uni, exp_uni, i_input_bi, exp_bi
            );
            if (code == 6) begin
                // A short column would be zero-extended by %b and pass, so reject it here.
                if (token_len(tok_scale) != `GEN_SCALE_W) begin
                    $display("FAIL cycle %0d: scale column is %0d bits, port is %0d",
                             n, token_len(tok_scale), `GEN_SCALE_W);
                    fails = fails + 1;
                end
                i_scale = token_bits(tok_scale);
                if (rst_flag) begin
                    i_rst_n = 1'b0;
                    @(posedge i_clk);   // async reset fires here (acc <= 0)
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                #1;                     // let the combinational outputs settle
                n = n + 1;
                if (o_uni !== exp_uni) begin
                    $display("FAIL cycle %0d unipolar: i_input=%b i_scale=%b got %b exp %b",
                             n, i_input_uni, i_scale, o_uni, exp_uni);
                    fails = fails + 1;
                end
                if (o_bi !== exp_bi) begin
                    $display("FAIL cycle %0d bipolar: i_input=%b i_scale=%b got %b exp %b",
                             n, i_input_bi, i_scale, o_bi, exp_bi);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the accumulator state
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS add_scale_dyn: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL add_scale_dyn: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
