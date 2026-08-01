`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "signabs/vec/signabs_params.vh"
// Python golden sign/abs use acc_next from the current input. Outputs are
// checked before the posedge commits acc_next; reset loads ACC_MED.
// Co-sim: make test OP=signabs
module signabs_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire sign_bit;
    wire abs_bit;

    signabs #(.WIDTH(`GEN_WIDTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in_bit),
        .o_sign  (sign_bit),
        .o_abs   (abs_bit)
    );

    integer fd, code, n, fails;
    reg rflag, a, exp_sign, exp_abs;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across a clock edge to load acc = ACC_MED, then
        // settle on a negedge with reset released so the first checked state is
        // the exact post-reset() accumulator.
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);

        fd = $fopen("vec/signabs.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/signabs.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%b %b %b %b\n", rflag, a, exp_sign, exp_abs);
            if (code == 4) begin
                if (rflag) begin
                    rst_n = 1'b0;
                    @(posedge clk);   // async reset loads acc = ACC_MED
                    @(negedge clk);
                    rst_n = 1'b1;
                end
                in_bit = a;
                #1;
                n = n + 1;
                if (sign_bit !== exp_sign || abs_bit !== exp_abs) begin
                    $display("FAIL cyc=%0d in=%b : got sign=%b abs=%b exp sign=%b abs=%b",
                             n, a, sign_bit, abs_bit, exp_sign, exp_abs);
                    fails = fails + 1;
                end
                @(posedge clk);
                @(negedge clk);
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS signabs: %0d/%0d vectors", n, n);
        else
            $display("FAIL signabs: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
