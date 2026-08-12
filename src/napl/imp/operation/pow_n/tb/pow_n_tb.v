`timescale 1ns/1ps
`default_nettype none
// Generated N mirrors the Python model config.
`include "pow_n/vec/pow_n_params.vh"
// Python golden rows are <rst> <input> <unipolar> <bipolar>. Both polarity DUTs
// share the one input stream; outputs are checked before the posedge advances
// the dff chain, and rst=1 first clears it.
// Co-sim: make test OP=pow_n


module pow_n_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output_uni, o_output_bi;

    // Both polarities use the Python model's generated N.
    pow_n_unipolar #(.N(`GEN_N)) dut_uni (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output_uni)
    );
    pow_n_bipolar #(.N(`GEN_N)) dut_bi (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output_bi)
    );

    integer fd, code, n, fails;
    reg rst_s, in_s, exp_uni, exp_bi;

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    initial begin
        i_input = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;

        // Reset clears the dff chain across a posedge.
        i_rst_n = 1'b0;
        @(posedge i_clk);
        #1 i_rst_n = 1'b1;

        fd = $fopen("vec/pow_n.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/pow_n.vec (run `make vectors` first)");
            $finish;
        end

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_s, in_s, exp_uni, exp_bi);
            if (code == 4) begin
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
                @(posedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS pow_n: %0d/%0d vectors", n, n);
        else
            $display("FAIL pow_n: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
