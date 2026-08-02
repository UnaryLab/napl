`timescale 1ns/1ps
`default_nettype none
`include "div_gaines/vec/div_gaines_params.vh"


module div_gaines_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_dividend_uni;
    reg i_divisor_uni;
    reg i_dividend_bi;
    reg i_divisor_bi;
    wire o_out_uni;
    wire o_out_bi;

    div_gaines_unipolar #(.DEPTH(`GEN_DEPTH)) dut_uni (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_dividend(i_dividend_uni),
        .i_divisor(i_divisor_uni),
        .o_out(o_out_uni)
    );
    div_gaines_bipolar #(.DEPTH(`GEN_DEPTH)) dut_bi (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_dividend(i_dividend_bi),
        .i_divisor(i_divisor_bi),
        .o_out(o_out_bi)
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
        i_dividend_uni = 1'b0;
        i_divisor_uni = 1'b0;
        i_dividend_bi = 1'b0;
        i_divisor_bi = 1'b0;

        if (`GEN_PP_DELAY != 1) begin
            $display("ERROR: div_gaines pp_delay must be 1");
            $finish;
        end

        fd = $fopen("vec/div_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/div_gaines.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %b %b %b %b\n",
                reset_flag,
                i_dividend_uni, i_divisor_uni, expected_uni,
                i_dividend_bi, i_divisor_bi, expected_bi
            );
            if (code == 7) begin
                if (reset_flag)
                    reset_duts;
                #1;
                count = count + 1;
                if (o_out_uni !== expected_uni) begin
                    $display(
                        "FAIL div_gaines unipolar cycle %0d: got=%b expected=%b",
                        count, o_out_uni, expected_uni
                    );
                    fails = fails + 1;
                end
                if (o_out_bi !== expected_bi) begin
                    $display(
                        "FAIL div_gaines bipolar cycle %0d: got=%b expected=%b",
                        count, o_out_bi, expected_bi
                    );
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS div_gaines: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL div_gaines: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
