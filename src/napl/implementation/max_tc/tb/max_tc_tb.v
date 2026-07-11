`timescale 1ns/1ps
`default_nettype none
//==============================================================================
// Self-checking testbench for max_tc.
//
// Reads golden vectors produced by gen/gen_max_tc.py (from the napl Python
// model) and asserts the RTL reproduces them. Prints "PASS ..." iff every
// vector matches; the Makefile greps for that line to decide the exit status.
//
// Run (from src/napl/implementation/):
//   make test OP=max_tc
//==============================================================================
module max_tc_tb;
    reg  i_in_0, i_in_1;
    wire o_out;

    max_tc dut (.i_in_0(i_in_0), .i_in_1(i_in_1), .o_out(o_out));

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
                i_in_0 = a;
                i_in_1 = b;
                #1;                         // let the combinational logic settle
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
