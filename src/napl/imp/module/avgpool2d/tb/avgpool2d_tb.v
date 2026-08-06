`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "avgpool2d/vec/avgpool2d_params.vh"
// Python golden rows are <rst> <in_bits> <out_bits> <out_d8_bits>, one per timestep.
// Each vector column is scanned as text and its character count is checked against
// the port width before it is converted to bits, so a row that is not exactly as
// wide as its port fails here instead of being zero-extended silently by %b.
// Outputs are combinational in the arrival cycle, so each row is checked before the
// posedge that advances the lane accumulators. rst=1 pulses i_rst_n low first.
// Co-sim: make test MODULE=avgpool2d


module avgpool2d_tb;
    localparam integer IN_WIDTH = `GEN_LANES * `GEN_KERNEL_AREA;

    reg                 i_clk;
    reg                 i_rst_n;
    reg  [IN_WIDTH-1:0] i_input_spike;
    wire [`GEN_LANES-1:0] o_out;
    wire [`GEN_LANES-1:0] o_out_d8;

    avgpool2d #(
        .KERNEL_AREA (`GEN_KERNEL_AREA),
        .DIVISOR     (`GEN_DIVISOR),
        .LANES       (`GEN_LANES)
    ) dut (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (i_input_spike),
        .o_out         (o_out)
    );

    // Second elaboration proves DIVISOR is independent of the window area.
    avgpool2d #(
        .KERNEL_AREA (`GEN_KERNEL_AREA),
        .DIVISOR     (`GEN_DIVISOR_D8),
        .LANES       (`GEN_LANES)
    ) dut_d8 (
        .i_clk         (i_clk),
        .i_rst_n       (i_rst_n),
        .i_input_spike (i_input_spike),
        .o_out         (o_out_d8)
    );

    // One character wider than the widest golden column: $fscanf("%s") truncates
    // to the token width, so a column read into a reg exactly as wide as it should
    // be saturates at the expected length and an over-long column would pass the
    // width check. The spare character makes an over-long column read back long.
    localparam integer MAX_CHARS = IN_WIDTH + 1;

    integer fd, code, n, fails;
    reg                   rst;
    reg  [IN_WIDTH-1:0]   in_bits;
    reg  [`GEN_LANES-1:0] exp_out;
    reg  [`GEN_LANES-1:0] exp_out_d8;
    reg  [1023:0]         hdr_line;
    reg  [MAX_CHARS*8-1:0] tok_in;
    reg  [MAX_CHARS*8-1:0] tok_out;
    reg  [MAX_CHARS*8-1:0] tok_out_d8;


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


    // A short column would be zero-extended by %b and pass, so reject it here.
    task check_width;
        input [MAX_CHARS*8-1:0] token;
        input integer expected;
        input [127:0] label;
        begin
            if (token_len(token) != expected) begin
                $display("FAIL n=%0d %0s: golden column is %0d bits, port is %0d",
                         n, label, token_len(token), expected);
                fails = fails + 1;
            end
        end
    endtask

    initial begin
        i_clk         = 1'b0;
        i_rst_n       = 1'b1;
        i_input_spike = {IN_WIDTH{1'b0}};

        fd = $fopen("vec/avgpool2d.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/avgpool2d.vec (run `make vectors` first)");
            $finish;
        end

        // skip the header line
        code = $fgets(hdr_line, fd);

        n = 0;
        fails = 0;
        // Each row is compared before the posedge, so this testbench can only drive a
        // pp_delay of 0; that pre-posedge sampling is what enforces it. This check
        // rejects a model whose pp_delay stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL avgpool2d: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // The loop ends on the first row that does not yield all 4 columns, so a
        // scan that stops consuming ends the run instead of spinning on $feof.
        code = $fscanf(fd, "%d %s %s %s\n", rst, tok_in, tok_out, tok_out_d8);
        while (code == 4) begin
            begin : g_row
                check_width(tok_in, IN_WIDTH, "in_bits");
                check_width(tok_out, `GEN_LANES, "out_bits");
                check_width(tok_out_d8, `GEN_LANES, "out_d8_bits");
                in_bits    = token_bits(tok_in);
                exp_out    = token_bits(tok_out);
                exp_out_d8 = token_bits(tok_out_d8);

                // reset boundary: clear every lane accumulator before this timestep
                if (rst == 1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                i_input_spike = in_bits;
                #1;

                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL n=%0d divisor=%0d : got %b exp %b",
                             n, `GEN_DIVISOR, o_out, exp_out);
                    fails = fails + 1;
                end
                if (o_out_d8 !== exp_out_d8) begin
                    $display("FAIL n=%0d divisor=%0d : got %b exp %b",
                             n, `GEN_DIVISOR_D8, o_out_d8, exp_out_d8);
                    fails = fails + 1;
                end

                // clock edge advances the lane accumulators for the next timestep
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end

            code = $fscanf(fd, "%d %s %s %s\n", rst, tok_in, tok_out, tok_out_d8);
        end
        $fclose(fd);

        // A vec file that lost or gained rows would otherwise pass on the rows it
        // still holds, so the row count is checked against the generator's. The
        // check waits on gen_avgpool2d.py emitting GEN_VECTORS.
`ifdef GEN_VECTORS
        if (n != `GEN_VECTORS) begin
            $display("FAIL avgpool2d: consumed %0d vectors, generator wrote %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end
`endif

        if (fails == 0)
            $display("PASS avgpool2d: %0d/%0d vectors (%0d lanes, divisor %0d and %0d)",
                     n, n, `GEN_LANES, `GEN_DIVISOR, `GEN_DIVISOR_D8);
        else
            $display("FAIL avgpool2d: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
