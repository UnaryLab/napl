`timescale 1ns/1ps
`default_nettype none
`include "sqrt_gaines/vec/sqrt_gaines_params.vh"


module sqrt_gaines_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_input_uni;
    reg i_input_bi;
    wire o_output_uni;
    wire o_output_bi;

    sqrt_gaines_unipolar #(.WIDTH(`GEN_WIDTH)) dut_uni (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input_uni),
        .o_output(o_output_uni)
    );
    sqrt_gaines_bipolar #(.WIDTH(`GEN_WIDTH)) dut_bi (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input_bi),
        .o_output(o_output_bi)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    reg reset_flag;
    reg expected_uni;
    reg expected_bi;

    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;


    task reset_duts;
        begin
            i_rst_n = 1'b0;
            @(posedge i_clk);
            @(negedge i_clk);
            i_rst_n = 1'b1;
        end
    endtask


    initial begin
        i_rst_n = 1'b1;
        i_input_uni = 1'b0;
        i_input_bi = 1'b0;

        if (`GEN_PP_DELAY != 1) begin
            $display("ERROR: sqrt_gaines pp_delay must be 1");
            $finish;
        end

        fd = $fopen("vec/sqrt_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sqrt_gaines.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %b %b\n",
                reset_flag, i_input_uni, expected_uni, i_input_bi, expected_bi
            );
            if (code == 5) begin
                if (reset_flag)
                    reset_duts;
                #1;
                count = count + 1;
                if (o_output_uni !== expected_uni) begin
                    $display(
                        "FAIL sqrt_gaines unipolar cycle %0d: got=%b expected=%b",
                        count, o_output_uni, expected_uni
                    );
                    fails = fails + 1;
                end
                if (o_output_bi !== expected_bi) begin
                    $display(
                        "FAIL sqrt_gaines bipolar cycle %0d: got=%b expected=%b",
                        count, o_output_bi, expected_bi
                    );
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sqrt_gaines: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL sqrt_gaines: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
