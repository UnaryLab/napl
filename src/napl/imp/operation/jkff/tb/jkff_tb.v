`timescale 1ns/1ps
`default_nettype none
`include "jkff/vec/jkff_params.vh"
// Python golden output is checked after each posedge; R clears q before replay.
// Co-sim: make test OP=jkff


module jkff_tb;
    reg  i_clk, i_rst_n, i_input_j, i_input_k;
    wire o_q;

    jkff dut (
        .i_clk(i_clk),
        .i_rst_n(i_rst_n),
        .i_input_j(i_input_j),
        .i_input_k(i_input_k),
        .o_q(o_q)
    );

    // 10 ns clock period.
    initial i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    integer fd, code, n, fails;
    reg [8*8-1:0] tok;
    reg j, k, exp_q;

    initial begin
        fd = $fopen("vec/jkff.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/jkff.vec (run `make vectors` first)");
            $finish;
        end

        // Reset to the model's post-reset() state (q = 0).
        i_input_j = 1'b0;
        i_input_k = 1'b0;
        i_rst_n   = 1'b0;
        @(posedge i_clk);
        #1 i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != 1) begin
            $display("FAIL jkff: observed latency 1, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s", tok);
            if (code != 1) begin
                code = 0;
            end else if (tok == "R") begin
                i_rst_n   = 1'b0;
                i_input_j = 1'b0;
                i_input_k = 1'b0;
                @(posedge i_clk);
                @(negedge i_clk);
                i_rst_n = 1'b1;
            end else begin
                j = (tok[7:0] == "1");
                code = $fscanf(fd, "%b %b\n", k, exp_q);
                // Drive inputs before the edge that updates the register.
                @(negedge i_clk);
                i_input_j = j;
                i_input_k = k;
                @(posedge i_clk);   // one forward() timestep
                #1;                 // let the registered output settle
                n = n + 1;
                if (o_q !== exp_q) begin
                    $display("FAIL t=%0d j=%b k=%b : got %b exp %b", n, j, k, o_q, exp_q);
                    fails = fails + 1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS jkff: %0d/%0d vectors", n, n);
        else
            $display("FAIL jkff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
