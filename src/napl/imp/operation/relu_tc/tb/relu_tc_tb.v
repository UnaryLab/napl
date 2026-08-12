`timescale 1ns/1ps
`default_nettype none
`include "relu_tc/vec/relu_tc_params.vh"

// Self-checking co-simulation testbench for relu_tc. The output is combinational
// in the arrival cycle, so it is checked before the clock edge advances the
// internal reference counter.

module relu_tc_tb;
    reg i_clk, i_rst_n, i_input;
    wire o_output;

    relu_tc #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_output(o_output)
    );

    integer fd, code, n, fails;
    reg rst, expected;

    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;


    task reset_dut;
        begin
            i_rst_n = 1'b0;
            @(posedge i_clk);
            @(negedge i_clk);
            i_rst_n = 1'b1;
        end
    endtask


    initial begin
        i_rst_n = 1'b1;
        i_input = 1'b0;
        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL relu_tc: expected pp_delay 0");
            $finish;
        end
        reset_dut;

        fd = $fopen("vec/relu_tc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/relu_tc.vec");
            $finish;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst, i_input, expected);
            if (code == 3) begin
                if (rst)
                    reset_dut;
                #1;
                n = n + 1;
                if (o_output !== expected) begin
                    $display("FAIL cycle=%0d in=%b got=%b exp=%b", n, i_input, o_output, expected);
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);
        if (fails == 0)
            $display("PASS relu_tc: %0d/%0d vectors", n, n);
        else
            $display("FAIL relu_tc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
