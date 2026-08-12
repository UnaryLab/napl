`timescale 1ns/1ps
`default_nettype none
`include "wta/vec/wta_params.vh"
// Golden rows are <rst> <input> <output>, produced by the Python model, with
// `input` packing the stacked streams. The output is combinational, so it is
// checked before the posedge that advances the history and winner latch; a
// marked row resets both first.
// Co-sim: make test OP=wta


module wta_tb;
    reg                   i_clk;
    reg                   i_rst_n;
    reg  [`GEN_ENTRY-1:0] i_input;
    wire                  o_output;

    wta #(
        .ENTRY(`GEN_ENTRY)
    ) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input (i_input),
        .o_output(o_output)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg rst_flag, exp_out;

    initial begin
        fd = $fopen("vec/wta.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/wta.vec (run `make vectors` first)");
            $finish;
        end

        i_input = {`GEN_ENTRY{1'b1}};
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // The output is a gate network over the current input and the held state.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL wta: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b\n", rst_flag, i_input, exp_out);
            if (code == 3) begin
                if (rst_flag) begin
                    i_rst_n = 1'b0;
                    @(posedge i_clk);   // the async reset reloads previous and fired here
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end
                #1;                     // let the combinational output settle
                n = n + 1;
                if (o_output !== exp_out) begin
                    $display("FAIL cycle %0d: i_input=%b got %b exp %b",
                             n, i_input, o_output, exp_out);
                    fails = fails + 1;
                end
                @(posedge i_clk);       // advance the history and the winner latch
                @(negedge i_clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS wta: %0d/%0d vectors", n, n);
        else
            $display("FAIL wta: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
