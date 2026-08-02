`timescale 1ns/1ps
`default_nettype none
`include "sync_skewed_int/vec/sync_skewed_int_params.vh"


module sync_skewed_int_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_input_1;
    reg i_input_2;
    wire [`GEN_WIDTH-1:0] o_out_1;
    wire o_out_2;

    sync_skewed_int #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input_1(i_input_1),
        .i_input_2(i_input_2),
        .o_out_1(o_out_1),
        .o_out_2(o_out_2)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    integer expected_1;
    reg reset_flag;
    reg expected_2;

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
        i_input_1 = 1'b0;
        i_input_2 = 1'b0;
        i_rst_n = 1'b1;

        if (`GEN_PP_DELAY != 0) begin
            $display("ERROR: sync_skewed_int pp_delay must be 0");
            $finish;
        end

        fd = $fopen("vec/sync_skewed_int.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sync_skewed_int.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %d %b\n",
                reset_flag, i_input_1, i_input_2, expected_1, expected_2
            );
            if (code == 5) begin
                if (reset_flag)
                    reset_dut;
                #1;
                count = count + 1;
                if (o_out_1 !== expected_1[`GEN_WIDTH-1:0]) begin
                    $display(
                        "FAIL sync_skewed_int cycle %0d output 1: got=%0d expected=%0d",
                        count, o_out_1, expected_1
                    );
                    fails = fails + 1;
                end
                if (o_out_2 !== expected_2) begin
                    $display(
                        "FAIL sync_skewed_int cycle %0d output 2: got=%b expected=%b",
                        count, o_out_2, expected_2
                    );
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display(
                "PASS sync_skewed_int: %0d/%0d vectors",
                count, count
            );
        else
            $display(
                "FAIL sync_skewed_int: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
