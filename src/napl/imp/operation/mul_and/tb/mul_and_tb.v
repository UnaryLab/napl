`timescale 1ns/1ps
`default_nettype none
// Python golden vectors check both polarity variants.
// Co-sim: make test OP=mul_and


module mul_and_tb;
    reg  in_0, in_1;
    wire out_uni, out_bi;

    mul_and_unipolar dut_uni (.i_input_0(in_0), .i_input_1(in_1), .o_out(out_uni));
    mul_and_bipolar  dut_bi  (.i_input_0(in_0), .i_input_1(in_1), .o_out(out_bi));

    integer fd, code, n, fails;
    reg a, b, exp_uni, exp_bi;

    initial begin
        fd = $fopen("vec/mul_and.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_and.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", a, b, exp_uni, exp_bi);
            if (code == 4) begin
                in_0 = a;
                in_1 = b;
                #1;
                n = n + 1;
                if (out_uni !== exp_uni) begin
                    $display("FAIL[uni] in_0=%b in_1=%b : got %b exp %b", a, b, out_uni, exp_uni);
                    fails = fails + 1;
                end
                if (out_bi !== exp_bi) begin
                    $display("FAIL[bi]  in_0=%b in_1=%b : got %b exp %b", a, b, out_bi, exp_bi);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS mul_and: %0d/%0d vectors", n, n);
        else
            $display("FAIL mul_and: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
