`timescale 1ns/1ps
`default_nettype none
`include "tanh_pn/vec/tanh_pn_params.vh"


module tanh_pn_tb;
    reg i_clk;
    reg i_rst_n;
    reg i_input;
    wire o_out;

    tanh_pn #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input(i_input),
        .o_out(o_out)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    reg [8*8-1:0] token;
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
        i_input = 1'b0;
        i_rst_n = 1'b1;
        reset_dut;

        if (`GEN_PP_DELAY != 1) begin
            $display("ERROR: tanh_pn pp_delay must be 1");
            $finish;
        end

        fd = $fopen("vec/tanh_pn.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/tanh_pn.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s", token);
            if (code == 1 && token == "R") begin
                reset_dut;
            end else if (code == 1) begin
                i_input = (token[7:0] == "1");
                code = $fscanf(fd, "%b\n", expected);
                #1;
                count = count + 1;
                if (o_out !== expected) begin
                    $display(
                        "FAIL tanh_pn cycle %0d: in=%b got=%b expected=%b",
                        count, i_input, o_out, expected
                    );
                    fails = fails + 1;
                end
                @(posedge i_clk);
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS tanh_pn: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL tanh_pn: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
