`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH/WIDTH mirror the Python model configuration.
`include "div_cordiv/vec/div_cordiv_params.vh"
// Python golden output is checked before each posedge advances buffer and index.
// R requests an active-low reset before replay continues.
// Co-sim: make test OP=div_cordiv
module div_cordiv_tb;
    reg  clk, rst_n;
    reg  dividend, divisor;
    wire quotient;

    div_cordiv #(
        .DEPTH(`GEN_DEPTH),
        .WIDTH(`GEN_WIDTH)
    ) dut (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_dividend(dividend),
        .i_divisor(divisor),
        .o_quotient(quotient)
    );

    integer fd, code, n, fails;
    reg [7:0] tag;
    reg dv, ds, exp_q;

    task do_reset;
        begin
            rst_n = 1'b0;
            clk = 1'b1; #1;   // async reset fires here (buffer=0, idx=0)
            clk = 1'b0; #1;
            rst_n = 1'b1;
            #1;
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b1;
        dividend = 1'b0;
        divisor = 1'b0;
        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != 0) begin
            $display(
                "FAIL div_cordiv: observed latency 0, expected pp_delay %0d",
                `GEN_PP_DELAY
            );
            $finish;
        end

        do_reset;

        fd = $fopen("vec/div_cordiv.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/div_cordiv.vec (run `make vectors` first)");
            $finish;
        end

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s", tag);
            if (code == 1) begin
                if (tag == "R") begin
                    do_reset;
                end else begin
                    dv = (tag == "1");
                    code = $fscanf(fd, "%b %b\n", ds, exp_q);
                    if (code == 2) begin
                        dividend = dv;
                        divisor  = ds;
                        #1;                  // let the combinational quotient settle
                        n = n + 1;
                        if (quotient !== exp_q) begin
                            $display("FAIL cyc=%0d dividend=%b divisor=%b : got %b exp %b",
                                     n, dv, ds, quotient, exp_q);
                            fails = fails + 1;
                        end
                        clk = 1'b1; #1;
                        clk = 1'b0; #1;
                    end
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS div_cordiv: %0d/%0d vectors", n, n);
        else
            $display("FAIL div_cordiv: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
