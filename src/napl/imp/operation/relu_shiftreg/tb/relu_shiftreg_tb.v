`timescale 1ns/1ps
`default_nettype none
`include "relu_shiftreg/vec/relu_shiftreg_params.vh"

// Self-checking co-simulation testbench for relu_shiftreg. Each row is one
// scalar Python forward() timestep; reset rows pulse the active-low reset first.
module relu_shiftreg_tb;
    reg i_clk, i_rst_n, i_input;
    wire o_out;

    relu_shiftreg #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input), .o_out(o_out)
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
            $display("FAIL relu_shiftreg: expected pp_delay 0");
            $finish;
        end
        reset_dut;

        fd = $fopen("vec/relu_shiftreg.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/relu_shiftreg.vec");
            $finish;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst, i_input, expected);
            if (code == 3) begin
                if (rst)
                    reset_dut;
                #1;
                n = n + 1;
                if (o_out !== expected) begin
                    $display("FAIL cycle=%0d in=%b got=%b exp=%b", n, i_input, o_out, expected);
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);
        if (fails == 0)
            $display("PASS relu_shiftreg: %0d/%0d vectors", n, n);
        else
            $display("FAIL relu_shiftreg: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
