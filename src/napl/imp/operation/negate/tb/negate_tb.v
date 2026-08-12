`timescale 1ns/1ps
`default_nettype none
`include "negate/vec/negate_params.vh"
// Golden rows are <input> <output>, produced by the Python model. negate is
// combinational, so each row is checked after the input settles.
// Co-sim: make test OP=negate


module negate_tb;
    reg  i_input;
    wire o_output;

    negate dut (
        .i_input (i_input),
        .o_output(o_output)
    );

    integer fd, code, n, fails;
    reg exp_out;

    initial begin
        fd = $fopen("vec/negate.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/negate.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // The circuit is a wire, so the model's zero pp_delay is the only legal value.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL negate: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b\n", i_input, exp_out);
            if (code == 2) begin
                #1;
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL row %0d: i_input=%b got %b exp %b",
                             n, i_input, o_output, exp_out);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS negate: %0d/%0d vectors", n, n);
        else
            $display("FAIL negate: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
