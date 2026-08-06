`timescale 1ns/1ps
`default_nettype none
// Generated sizing mirrors the Python model configuration.
`include "avgpool2d/vec/avgpool2d_params.vh"
// Python golden rows are <rst> <in_bits> <out_bits> <out_d8_bits>, one per timestep.
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

    integer fd, code, n, fails;
    reg                   rst;
    reg  [IN_WIDTH-1:0]   in_bits;
    reg  [`GEN_LANES-1:0] exp_out;
    reg  [`GEN_LANES-1:0] exp_out_d8;
    reg  [1023:0]         hdr_line;

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

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%d %b %b %b\n", rst, in_bits, exp_out, exp_out_d8);
            if (code == 4) begin
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
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS avgpool2d: %0d/%0d vectors (%0d lanes, divisor %0d and %0d)",
                     n, n, `GEN_LANES, `GEN_DIVISOR, `GEN_DIVISOR_D8);
        else
            $display("FAIL avgpool2d: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
