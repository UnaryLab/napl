`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "sub_scale/vec/sub_scale_params.vh"
// Golden rows are <rst> <a> <b> <output>, produced by the Python model. sub_scale
// carries the inner add_scale accumulator, so reset precedes the marked row and
// the output is checked before the posedge advances state.
// Co-sim: make test OP=sub_scale


module sub_scale_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_0;
    reg  i_input_1;
    wire o_output;

    sub_scale #(
        .SCALE(`GEN_SCALE),
        .WIDTH(`GEN_WIDTH)
    ) dut (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input_0(i_input_0),
        .i_input_1(i_input_1),
        .o_output (o_output)
    );

    // 10ns clock
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_flag, exp_out;

    initial begin
        fd = $fopen("vec/sub_scale.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/sub_scale.vec (run `make vectors` first)");
            $finish;
        end

        i_input_0 = 1'b0;
        i_input_1 = 1'b0;
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // The path is negate (combinational) into add_scale (combinational output),
        // so the model's zero pp_delay is the only legal value.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL sub_scale: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rst_flag, i_input_0, i_input_1, exp_out);
            if (code == 4) begin
                if (rst_flag) begin
                    i_rst_n = 1'b0;
                    @(posedge i_clk);   // async reset fires here (acc <= 0)
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                #1;                     // let the combinational output settle
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL cycle %0d: a=%b b=%b got %b exp %b",
                             n, i_input_0, i_input_1, o_output, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the accumulator state
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS sub_scale: %0d/%0d vectors", n, n);
        else
            $display("FAIL sub_scale: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
