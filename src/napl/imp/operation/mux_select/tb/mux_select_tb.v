`timescale 1ns/1ps
`default_nettype none
`include "mux_select/vec/mux_select_params.vh"
// Golden rows are <select> <input_a> <input_b> <output>, from the Python model.
// mux_select is combinational, so each row is checked after the inputs settle.
// Co-sim: make test OP=mux_select


module mux_select_tb;
    reg  i_select;
    reg  i_input_a;
    reg  i_input_b;
    wire o_output;

    mux_select dut (
        .i_select (i_select),
        .i_input_a(i_input_a),
        .i_input_b(i_input_b),
        .o_output (o_output)
    );

    integer fd, code, n, fails;
    reg exp_out;

    initial begin
        fd = $fopen("vec/mux_select.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mux_select.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // The circuit is a single mux, so the model's zero pp_delay is the only legal value.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL mux_select: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", i_select, i_input_a, i_input_b, exp_out);
            if (code == 4) begin
                #1;
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL row %0d: i_select=%b i_input_a=%b i_input_b=%b got %b exp %b",
                             n, i_select, i_input_a, i_input_b, o_output, exp_out);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS mux_select: %0d/%0d vectors", n, n);
        else
            $display("FAIL mux_select: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
