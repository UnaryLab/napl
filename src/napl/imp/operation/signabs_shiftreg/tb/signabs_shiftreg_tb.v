`timescale 1ns/1ps
`default_nettype none
`include "signabs_shiftreg/vec/signabs_shiftreg_params.vh"

// Self-checking co-simulation testbench for signabs_shiftreg.
module signabs_shiftreg_tb;
    reg i_clk, i_rst_n, i_input;
    wire o_sign, o_magnitude;

    signabs_shiftreg #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk(i_clk), .i_rst_n(i_rst_n), .i_input(i_input),
        .o_sign(o_sign), .o_magnitude(o_magnitude)
    );

    integer fd, code, n, fails;
    reg rst, expected_sign, expected_magnitude;

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
            $display("FAIL signabs_shiftreg: expected pp_delay 0");
            $finish;
        end
        reset_dut;

        fd = $fopen("vec/signabs_shiftreg.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/signabs_shiftreg.vec");
            $finish;
        end
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %b\n",
                rst, i_input, expected_sign, expected_magnitude
            );
            if (code == 4) begin
                if (rst)
                    reset_dut;
                #1;
                n = n + 1;
                if (o_sign !== expected_sign || o_magnitude !== expected_magnitude) begin
                    $display(
                        "FAIL cycle=%0d in=%b got sign=%b mag=%b exp sign=%b mag=%b",
                        n, i_input, o_sign, o_magnitude, expected_sign, expected_magnitude
                    );
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);
        if (fails == 0)
            $display("PASS signabs_shiftreg: %0d/%0d vectors", n, n);
        else
            $display("FAIL signabs_shiftreg: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
