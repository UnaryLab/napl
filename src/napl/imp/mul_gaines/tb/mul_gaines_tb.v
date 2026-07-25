`timescale 1ns/1ps
`default_nettype none

module mul_gaines_tb;
    reg i_in_0;
    reg i_in_1;
    wire o_uni;
    wire o_bi;

    mul_gaines_unipolar dut_uni (
        .i_in_0(i_in_0),
        .i_in_1(i_in_1),
        .o_out(o_uni)
    );
    mul_gaines_bipolar dut_bi (
        .i_in_0(i_in_0),
        .i_in_1(i_in_1),
        .o_out(o_bi)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    reg exp_uni;
    reg exp_bi;

    initial begin
        fd = $fopen("vec/mul_gaines.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_gaines.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b %b\n",
                i_in_0, i_in_1, exp_uni, exp_bi
            );
            if (code == 4) begin
                #1;
                count = count + 1;
                if (o_uni !== exp_uni)
                    fails = fails + 1;
                if (o_bi !== exp_bi)
                    fails = fails + 1;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS mul_gaines: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL mul_gaines: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire
