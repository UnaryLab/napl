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
        .o_magnitude   (abs_bit)
    );

    // LINE_BYTES sizes the line buffer, which holds LINE_BYTES bytes, so a
    // newline-terminated row carries at most LINE_BYTES-1 payload bytes, and
    // DATA_COLS is the column count every data row carries. Neither constant
    // can be set wrong and still let a malformed row through, because the
    // trailing sentinel makes the column check two-sided. A DATA_COLS set too
    // large fatals on the first clean row. A LINE_BYTES set too large costs
    // guard precision rather than correctness, because the length assertion
    // can then fire only at the buffer boundary instead of near the true row
    // width, so both constants are measured from the vec file rather than
    // guessed.
    localparam LINE_BYTES = 32;
    localparam DATA_COLS  = 4;

    integer fd, code, chars, n, fails;
    reg rflag, a, exp_sign, exp_abs;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rflag;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_exp_sign;
    reg [8*8-1:0] tok_exp_abs;
    reg [8*LINE_BYTES-1:0] line;

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
        // A line carrying any other column count, a line that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "signabs: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s", tok_rflag, tok_a, tok_exp_sign, tok_exp_abs, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "signabs: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "signabs: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rflag !== "0" && tok_rflag !== "1")
                        $fatal(1, "signabs: row %0d input field rflag is \"%0s\", expected a single 0 or 1", n + 1, tok_rflag);
                    rflag = (tok_rflag == "1");
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "signabs: row %0d input field a is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_sign !== "0" && tok_exp_sign !== "1")
                        $fatal(1, "signabs: row %0d expected field exp_sign is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_sign);
                    exp_sign = (tok_exp_sign == "1");
                    if (tok_exp_abs !== "0" && tok_exp_abs !== "1")
                        $fatal(1, "signabs: row %0d expected field exp_abs is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_abs);
                    exp_abs = (tok_exp_abs == "1");
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
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL signabs: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS signabs: %0d/%0d vectors", n, n);
        else
            $display("FAIL signabs: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
