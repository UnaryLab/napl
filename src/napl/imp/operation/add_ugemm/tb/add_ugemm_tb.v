`timescale 1ns/1ps
`default_nettype none
`include "add_ugemm/vec/add_ugemm_params.vh"


module add_ugemm_tb;
    reg i_clk;
    reg i_rst_n;
    reg [`GEN_ENTRY-1:0] i_scaled_uni;
    reg [`GEN_ENTRY-1:0] i_scaled_bi;
    reg [`GEN_ENTRY-1:0] i_unscaled_uni;
    reg [`GEN_ENTRY-1:0] i_unscaled_bi;
    wire o_scaled_uni;
    wire o_scaled_bi;
    wire o_unscaled_uni;
    wire o_unscaled_bi;

    add_ugemm_unipolar #(
        .SCALED(`GEN_SCALED),
        .ENTRY(`GEN_ENTRY),
        .COUNT_WIDTH(`GEN_COUNT_WIDTH),
        .ACC_WIDTH(`GEN_ACC_WIDTH)
    ) dut_scaled_uni (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_scaled_uni),
        .o_out(o_scaled_uni)
    );
    add_ugemm_bipolar #(
        .SCALED(`GEN_SCALED),
        .ENTRY(`GEN_ENTRY),
        .COUNT_WIDTH(`GEN_COUNT_WIDTH),
        .ACC_WIDTH(`GEN_ACC_WIDTH)
    ) dut_scaled_bi (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_scaled_bi),
        .o_out(o_scaled_bi)
    );
    add_ugemm_unipolar #(
        .SCALED(`GEN_UNSCALED),
        .ENTRY(`GEN_ENTRY),
        .COUNT_WIDTH(`GEN_COUNT_WIDTH),
        .ACC_WIDTH(`GEN_ACC_WIDTH)
    ) dut_unscaled_uni (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_unscaled_uni),
        .o_out(o_unscaled_uni)
    );
    add_ugemm_bipolar #(
        .SCALED(`GEN_UNSCALED),
        .ENTRY(`GEN_ENTRY),
        .COUNT_WIDTH(`GEN_COUNT_WIDTH),
        .ACC_WIDTH(`GEN_ACC_WIDTH)
    ) dut_unscaled_bi (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_unscaled_bi),
        .o_out(o_unscaled_bi)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    reg reset_flag;
    reg expected_scaled_uni;
    reg expected_scaled_bi;
    reg expected_unscaled_uni;
    reg expected_unscaled_bi;

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
        i_scaled_uni = {`GEN_ENTRY{1'b0}};
        i_scaled_bi = {`GEN_ENTRY{1'b0}};
        i_unscaled_uni = {`GEN_ENTRY{1'b0}};
        i_unscaled_bi = {`GEN_ENTRY{1'b0}};

        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL add_ugemm: observed latency 0, expected %0d", `GEN_PP_DELAY);
            $finish;
        end

        fd = $fopen("vec/add_ugemm.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/add_ugemm.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %b %b %b %b %b %b\n",
                reset_flag,
                i_scaled_uni, expected_scaled_uni,
                i_scaled_bi, expected_scaled_bi,
                i_unscaled_uni, expected_unscaled_uni,
                i_unscaled_bi, expected_unscaled_bi
            );
            if (code == 9) begin
                if (reset_flag)
                    reset_duts;
                #1;
                count = count + 1;
                if (o_scaled_uni !== expected_scaled_uni) begin
                    $display("FAIL add_ugemm scaled unipolar cycle %0d", count);
                    fails = fails + 1;
                end
                if (o_scaled_bi !== expected_scaled_bi) begin
                    $display("FAIL add_ugemm scaled bipolar cycle %0d", count);
                    fails = fails + 1;
                end
                if (o_unscaled_uni !== expected_unscaled_uni) begin
                    $display("FAIL add_ugemm unscaled unipolar cycle %0d", count);
                    fails = fails + 1;
                end
                if (o_unscaled_bi !== expected_unscaled_bi) begin
                    $display("FAIL add_ugemm unscaled bipolar cycle %0d", count);
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS add_ugemm: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL add_ugemm: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
