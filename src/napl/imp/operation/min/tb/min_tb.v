`timescale 1ns/1ps
`default_nettype none
`include "min/vec/min_params.vh"
// Python golden min/argmin outputs are checked before each posedge advances
// state. The reset flag clears result and cnt before its row.
// Co-sim: make test OP=min


module min_tb;
    reg  clk, rst_n;
    reg  in_0, in_1;
    wire o_output, o_index;

    min dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input_0  (in_0),
        .i_input_1  (in_1),
        .o_output (o_output),
        .o_index  (o_index)
    );

    // 10ns clock
    initial clk = 1'b0;
    always #5 clk = ~clk;

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
    localparam DATA_COLS  = 5;

    integer fd, code, chars, n, fails;
    reg a, b, exp_min, exp_arg, rst_mid;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_b;
    reg [8*8-1:0] tok_exp_min;
    reg [8*8-1:0] tok_exp_arg;
    reg [8*8-1:0] tok_rst_mid;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        fd = $fopen("vec/min.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/min.vec (run `make vectors` first)");
            $finish;
        end

        // assert reset (active low) -> post-reset() state
        in_0  = 1'b0;
        in_1  = 1'b0;
        rst_n = 1'b0;
        @(posedge clk);
        #1 rst_n = 1'b1;   // release reset shortly after the edge

        n     = 0;
        fails = 0;
        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "min: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s", tok_a, tok_b, tok_exp_min, tok_exp_arg, tok_rst_mid, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "min: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "min: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "min: row %0d input field in_0 is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    if (tok_b !== "0" && tok_b !== "1")
                        $fatal(1, "min: row %0d input field in_1 is \"%0s\", expected a single 0 or 1", n + 1, tok_b);
                    b = (tok_b == "1");
                    if (tok_rst_mid !== "0" && tok_rst_mid !== "1")
                        $fatal(1, "min: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_mid);
                    rst_mid = (tok_rst_mid == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_min !== "0" && tok_exp_min !== "1")
                        $fatal(1, "min: row %0d expected field exp_min is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_min);
                    exp_min = (tok_exp_min == "1");
                    if (tok_exp_arg !== "0" && tok_exp_arg !== "1")
                        $fatal(1, "min: row %0d expected field exp_arg is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_arg);
                    exp_arg = (tok_exp_arg == "1");
                    if (rst_mid) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        #1 rst_n = 1'b1;
                    end
                    in_0 = a;
                    in_1 = b;
                    #1;
                    n = n + 1;
                    if (o_output !== exp_min) begin
                        $display("FAIL[min] cyc=%0d in_0=%b in_1=%b : got %b exp %b",
                                 n, a, b, o_output, exp_min);
                        fails = fails + 1;
                    end
                    if (o_index !== exp_arg) begin
                        $display("FAIL[arg] cyc=%0d in_0=%b in_1=%b : got %b exp %b",
                                 n, a, b, o_index, exp_arg);
                        fails = fails + 1;
                    end
                    // advance one timestep: latch dff/cnt next-state
                    @(posedge clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL min: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS min: %0d/%0d vectors", n, n);
        else
            $display("FAIL min: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
