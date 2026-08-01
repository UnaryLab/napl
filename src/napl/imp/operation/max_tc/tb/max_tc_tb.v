`timescale 1ns/1ps
`default_nettype none
// Python golden vectors check the combinational output.
// Co-sim: make test OP=max_tc
module max_tc_tb;
    reg  i_input_0, i_input_1;
    wire o_out;

    max_tc dut (.i_input_0(i_input_0), .i_input_1(i_input_1), .o_out(o_out));

    integer fd, code, n, fails;
    reg a, b, exp_out;

    initial begin
        fd = $fopen("vec/max_tc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/max_tc.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", a, b, exp_out);
            if (code == 3) begin
                i_input_0 = a;
                i_input_1 = b;
                #1;
                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL in_0=%b in_1=%b : got %b exp %b", a, b, o_out, exp_out);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS max_tc: %0d/%0d vectors", n, n);
        else
            $display("FAIL max_tc: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
