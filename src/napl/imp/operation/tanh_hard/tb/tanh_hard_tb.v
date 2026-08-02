`timescale 1ns/1ps
`default_nettype none
// Python golden vectors check the zero-latency identity output.
// Co-sim: make test OP=tanh_hard


module tanh_hard_tb;
    reg  i_input;
    wire o_out;

    tanh_hard dut (.i_input(i_input), .o_out(o_out));

    integer fd, code, n, fails;
    reg in_b, exp_out;

    initial begin
        fd = $fopen("vec/tanh_hard.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/tanh_hard.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b\n", in_b, exp_out);
            if (code == 2) begin
                i_input = in_b;
                #1;
                n = n + 1;
                if (o_out !== exp_out) begin
                    $display("FAIL cycle %0d: i_input=%b got %b exp %b", n, in_b, o_out, exp_out);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS tanh_hard: %0d/%0d vectors", n, n);
        else
            $display("FAIL tanh_hard: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
