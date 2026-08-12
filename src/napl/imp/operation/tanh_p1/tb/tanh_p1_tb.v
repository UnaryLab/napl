`timescale 1ns/1ps
`default_nettype none
`include "tanh_p1/vec/tanh_p1_params.vh"


module tanh_p1_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_input;
    wire o_output;

    tanh_p1 #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input),
        .o_output(o_output)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    reg reset_flag;
    reg expected;

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

        if (`GEN_PP_DELAY != 0) begin
            $display("ERROR: tanh_p1 pp_delay must be 0");
            $finish;
        end

        fd = $fopen("vec/tanh_p1.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/tanh_p1.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b\n", reset_flag, i_input, expected
            );
            if (code == 3) begin
                if (reset_flag)
                    reset_dut;
                #1;
                count = count + 1;
                if (o_output !== expected) begin
                    $display(
                        "FAIL tanh_p1 cycle %0d: in=%b got=%b expected=%b",
                        count, i_input, o_output, expected
                    );
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS tanh_p1: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL tanh_p1: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
