`timescale 1ns/1ps
`default_nettype none
`include "mul_unibi/vec/mul_unibi_params.vh"
// Golden rows are <rst> <input_u> <input_b> <output>, produced by the Python
// model. The output is combinational, so it is checked before the posedge that
// advances the divider flip-flop; a marked row resets that flip-flop first.
// Co-sim: make test OP=mul_unibi


module mul_unibi_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_u;
    reg  i_input_b;
    wire o_output;

    mul_unibi dut (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input_u(i_input_u),
        .i_input_b(i_input_b),
        .o_output (o_output)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_flag, exp_out;

    initial begin
        fd = $fopen("vec/mul_unibi.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_unibi.vec (run `make vectors` first)");
            $finish;
        end

        i_input_u = 1'b0;
        i_input_b = 1'b0;
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // The output is a gate network over the current inputs and the held state.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL mul_unibi: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_flag, i_input_u, i_input_b, exp_out);
            if (code == 4) begin
                if (rst_flag) begin
                    i_rst_n = 1'b0;
                    @(posedge i_clk);   // the async reset clears q here
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                #1;                     // let the combinational output settle
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL cycle %0d: i_input_u=%b i_input_b=%b got %b exp %b",
                             n, i_input_u, i_input_b, o_output, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the divider flip-flop
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS mul_unibi: %0d/%0d vectors", n, n);
        else
            $display("FAIL mul_unibi: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
