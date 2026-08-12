`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH and SHIFT_DEPTH mirror the Python model configurations.
`include "square_dff/vec/square_dff_params.vh"
// Python golden rows are <rst> <input> <unipolar> <bipolar> <shift unipolar>
// <shift bipolar>. Outputs are checked before the posedge advances the delay;
// rst=1 first clears it.
// The SHIFT_DEPTH pair is the multi-cell elaboration: at DEPTH 1 the delay line
// holds one cell and its cell-to-cell shift is not generated.
// Co-sim: make test OP=square_dff


module square_dff_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output_uni, o_output_bi;
    wire o_output_shift_uni, o_output_shift_bi;

    // Both polarities use the Python model's generated DEPTH.
    square_dff_unipolar #(.DEPTH(`GEN_DEPTH)) dut_uni (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output_uni)
    );
    square_dff_bipolar #(.DEPTH(`GEN_DEPTH)) dut_bi (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output_bi)
    );
    square_dff_unipolar #(.DEPTH(`GEN_SHIFT_DEPTH)) dut_shift_uni (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output_shift_uni)
    );
    square_dff_bipolar #(.DEPTH(`GEN_SHIFT_DEPTH)) dut_shift_bi (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output_shift_bi)
    );

    integer fd, code, n, fails;
    reg rst_s, in_s, exp_uni, exp_bi, exp_shift_uni, exp_shift_bi;

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    initial begin
        i_input    = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;
        if (`GEN_PP_DELAY != 0) begin
            $display(
                "FAIL square_dff: observed latency 0, expected pp_delay %0d",
                `GEN_PP_DELAY
            );
            fails = fails + 1;
        end

        // Reset clears the delay line across a posedge.
        i_rst_n = 1'b0;
        @(posedge i_clk);
        #1 i_rst_n = 1'b1;

        fd = $fopen("vec/square_dff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/square_dff.vec (run `make vectors` first)");
            $finish;
        end

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b %b %b\n", rst_s, in_s, exp_uni, exp_bi,
                           exp_shift_uni, exp_shift_bi);
            if (code == 6) begin
                if (rst_s) begin
                    @(negedge i_clk);
                    i_rst_n = 1'b0;
                    @(posedge i_clk);
                    #1 i_rst_n = 1'b1;
                end
                @(negedge i_clk);
                i_input = in_s;
                #1;
                n = n + 1;
                if (o_output_uni !== exp_uni) begin
                    $display("FAIL[uni] t=%0d i_input=%b : got %b exp %b", n, in_s, o_output_uni, exp_uni);
                    fails = fails + 1;
                end
                if (o_output_bi !== exp_bi) begin
                    $display("FAIL[bi]  t=%0d i_input=%b : got %b exp %b", n, in_s, o_output_bi, exp_bi);
                    fails = fails + 1;
                end
                if (o_output_shift_uni !== exp_shift_uni) begin
                    $display("FAIL[shift uni] t=%0d i_input=%b : got %b exp %b", n, in_s, o_output_shift_uni, exp_shift_uni);
                    fails = fails + 1;
                end
                if (o_output_shift_bi !== exp_shift_bi) begin
                    $display("FAIL[shift bi]  t=%0d i_input=%b : got %b exp %b", n, in_s, o_output_shift_bi, exp_shift_bi);
                    fails = fails + 1;
                end
                @(posedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS square_dff: %0d/%0d vectors", n, n);
        else
            $display("FAIL square_dff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
