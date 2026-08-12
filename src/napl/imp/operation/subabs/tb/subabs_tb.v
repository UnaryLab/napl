`timescale 1ns/1ps
`default_nettype none
`include "subabs/vec/subabs_params.vh"
// Golden rows are <input_0> <input_1> <output>, produced by the Python model.
// subabs is combinational, so each row is checked after the inputs settle.
// Co-sim: make test OP=subabs


module subabs_tb;
    reg  i_input_0;
    reg  i_input_1;
    wire o_output;

    subabs dut (
        .i_input_0(i_input_0),
        .i_input_1(i_input_1),
        .o_output (o_output)
    );

    integer fd, code, n, fails;
    reg exp_out;

    initial begin
        fd = $fopen("vec/subabs.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/subabs.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // The circuit is a single gate, so the model's zero pp_delay is the only legal value.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL subabs: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", i_input_0, i_input_1, exp_out);
            if (code == 3) begin
                #1;
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL row %0d: i_input_0=%b i_input_1=%b got %b exp %b",
                             n, i_input_0, i_input_1, o_output, exp_out);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS subabs: %0d/%0d vectors", n, n);
        else
            $display("FAIL subabs: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
