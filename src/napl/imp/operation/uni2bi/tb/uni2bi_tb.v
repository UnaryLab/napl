`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "uni2bi/vec/uni2bi_params.vh"
// Python golden output is checked before each posedge updates acc. R requests
// active-low reset before replay continues.
// Co-sim: make test OP=uni2bi


module uni2bi_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    wire o_output;

    uni2bi #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input    (i_input),
        .o_output   (o_output)
    );

    integer fd, code, n, fails;
    reg [127:0] tok;
    reg in_bit, exp_out;

    // Reset clears acc across a clock edge.

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

        fd = $fopen("vec/uni2bi.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/uni2bi.vec (run `make vectors` first)");
            $finish;
        end

        do_reset;

        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s", tok);
            if (code == 1) begin
                if (tok == "R") begin
                    do_reset;
                end else begin
                    in_bit  = (tok == "1");
                    code    = $fscanf(fd, "%b", exp_out);
                    i_input    = in_bit;
                    #1;
                    n = n + 1;
                    if (o_output !== exp_out) begin
                        $display("FAIL cyc=%0d i_input=%b : got %b exp %b", n, in_bit, o_output, exp_out);
                        fails = fails + 1;
                    end
                    #1 i_clk = 1'b1; #1 i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS uni2bi: %0d/%0d vectors", n, n);
        else
            $display("FAIL uni2bi: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
