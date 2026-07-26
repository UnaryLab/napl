// Self-checking testbench for max_rc: replays vec/max_rc.vec cycle by cycle and
// compares o_max/o_arg to the golden columns from the napl Python model.
// Prints "PASS" only on a full bit-exact match.
`timescale 1ns / 1ps
`default_nettype none
`include "max_rc/vec/max_rc_params.vh"
module max_rc_tb;
    reg  clk;
    reg  rst_n;
    reg  in_0;
    reg  in_1;
    wire o_max;
    wire o_arg;

    integer fd, code, errors, count;
    integer v_reset, v_in_0, v_in_1, v_max, v_arg;
    reg [1023:0] line;

    max_rc dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input_0  (in_0),
        .i_input_1  (in_1),
        .o_max   (o_max),
        .o_arg   (o_arg)
    );

    // 10ns clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        errors = 0;
        count  = 0;
        in_0   = 1'b0;
        in_1   = 1'b0;

        // Pulse reset low so the DUT starts from the model's post-reset() state.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/max_rc.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/max_rc.vec");
            $finish;
        end

        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL max_rc: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            $finish;
        end

        // Drive each vector on the negedge, sample outputs (combinational) the
        // same cycle, then advance the posedge to update state.
        while (!$feof(fd)) begin
            code = $fgets(line, fd);
            if (code == 0) begin
                // skip
            end else begin
                code = $sscanf(
                    line, "%d %d %d %d %d",
                    v_reset, v_in_0, v_in_1, v_max, v_arg
                );
                if (code == 5) begin
                    @(negedge clk);
                    if (v_reset != 0) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    in_0 = v_in_0[0];
                    in_1 = v_in_1[0];
                    #1;  // let combinational outputs settle
                    if (o_max !== v_max[0] || o_arg !== v_arg[0]) begin
                        errors = errors + 1;
                        if (errors <= 10)
                            $display("MISMATCH t=%0d in=(%0d,%0d) got=(%b,%b) exp=(%0d,%0d)",
                                     count, v_in_0, v_in_1, o_max, o_arg, v_max, v_arg);
                    end
                    count = count + 1;
                    @(posedge clk);  // commit state for next cycle
                end
            end
        end
        $fclose(fd);

        if (errors == 0)
            $display("PASS max_rc: %0d vectors matched", count);
        else
            $display("FAIL max_rc: %0d/%0d mismatched", errors, count);
        $finish;
    end
endmodule
`default_nettype wire
