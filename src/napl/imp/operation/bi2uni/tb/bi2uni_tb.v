`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "bi2uni/vec/bi2uni_params.vh"
// Python golden output is checked before each posedge updates acc. R requests
// an active-low reset before replay continues.
// Co-sim: make test OP=bi2uni


module bi2uni_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_out;

    bi2uni #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input    (i_input),
        .o_out   (o_out)
    );

    integer fd, code, n, fails;
    reg [8*8-1:0] in_str, out_str;
    reg in_bit, exp_out;

    // pulse active-low reset to load acc = 0 (post-reset() state).

    task do_reset;
        begin
            i_rst_n = 1'b0;
            #1 i_clk = 1'b1; #1 i_clk = 1'b0;
            i_rst_n = 1'b1;
            #1;
        end
    endtask


    initial begin
        i_clk   = 1'b0;
        i_input    = 1'b0;
        i_rst_n = 1'b1;
        n       = 0;
        fails   = 0;

        fd = $fopen("vec/bi2uni.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/bi2uni.vec (run `make vectors` first)");
            $finish;
        end

        do_reset;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s %s\n", in_str, out_str);
            if (code == 2) begin
                if (in_str == "R") begin
                    do_reset;
                end else begin
                    in_bit  = (in_str == "1");
                    exp_out = (out_str == "1");
                    i_input = in_bit;
                    #1;                 // let the combinational output settle
                    n = n + 1;
                    if (o_out !== exp_out) begin
                        $display("FAIL cyc=%0d i_input=%b : got %b exp %b", n, in_bit, o_out, exp_out);
                        fails = fails + 1;
                    end
                    #1 i_clk = 1'b1; #1 i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS bi2uni: %0d/%0d vectors", n, n);
        else
            $display("FAIL bi2uni: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
