`timescale 1ns/1ps
`default_nettype none
// Generated sizes mirror the Python model configuration.
`include "decorr/vec/decorr_params.vh"
// Python golden rows are <rst> <in_0> <in_1> <out_0> <out_1>.
// Both outputs are read before the posedge that updates the buffers and the
// sequence counter; rst=1 opens a block by reloading the reset state.
// Co-sim: make test OP=decorr


module decorr_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_0;
    reg  i_input_1;
    wire o_output_0;
    wire o_output_1;

    decorr #(
        .DEPTH   (`GEN_DEPTH),
        .IDX_W   (`GEN_IDX_W),
        .SEQ_LEN (`GEN_SEQ_LEN),
        .SEQ_W   (`GEN_SEQ_W)
    ) dut (
        .i_clk     (i_clk),
        .i_rst_n   (i_rst_n),
        .i_input_0 (i_input_0),
        .i_input_1 (i_input_1),
        .o_output_0   (o_output_0),
        .o_output_1   (o_output_1)
    );

    integer fd, code, n, fails;
    reg rst, a, b, exp_0, exp_1;
    reg [1023:0] hdr_line;

    initial begin
        i_clk     = 1'b0;
        i_rst_n   = 1'b1;
        i_input_0 = 1'b0;
        i_input_1 = 1'b0;

        fd = $fopen("vec/decorr.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/decorr.vec (run `make vectors` first)");
            $finish;
        end

        // skip the header line
        code = $fgets(hdr_line, fd);

        n = 0;
        fails = 0;
        // Each row below is compared in the cycle its inputs are applied, with no
        // clock edge in between, so only a pp_delay of 0 can match the vectors.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL decorr: pp_delay %0d, but the vector rows compare at zero latency",
                     `GEN_PP_DELAY);
            fails = fails + 1;
        end

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b %b\n", rst, a, b, exp_0, exp_1);
            if (code == 5) begin
                // block boundary: reload the buffers and the counter before this cycle
                if (rst === 1'b1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                i_input_0 = a;
                i_input_1 = b;
                #1;

                n = n + 1;
                if (o_output_0 !== exp_0 || o_output_1 !== exp_1) begin
                    $display("FAIL n=%0d in=(%b,%b) : got (%b,%b) exp (%b,%b)",
                             n, a, b, o_output_0, o_output_1, exp_0, exp_1);
                    fails = fails + 1;
                end

                // clock edge stores this cycle's inputs and advances the counter
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS decorr: %0d/%0d vectors", n, n);
        else
            $display("FAIL decorr: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
