`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "relu_cnt/vec/relu_cnt_params.vh"
// Python golden output is checked before each posedge updates acc. The reset
// column requests active-low reset to HALF before its row.
// Co-sim: make test OP=relu_cnt


module relu_cnt_tb;
    localparam WIDTH = `GEN_WIDTH;

    reg  clk, rst_n, in;
    wire out;

    relu_cnt #(.WIDTH(WIDTH)) dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input    (in),
        .o_output   (out)
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
    localparam LINE_BYTES = 16;
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg a, exp_out, rst;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_exp_out;
    reg [8*8-1:0] tok_rst;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in    = 1'b0;
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/relu_cnt.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/relu_cnt.vec (run `make vectors` first)");
            $finish;
        end

        // Reset to the model's post-reset() state (acc = HALF). Hold rst_n low
        // across a posedge, then deassert on the falling edge so the first
        // driven vector sees acc = HALF (no stray advancing edge in between).
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "relu_cnt: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok_a, tok_exp_out, tok_rst, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "relu_cnt: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "relu_cnt: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "relu_cnt: row %0d input field in is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "relu_cnt: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "relu_cnt: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    // Mid-stream reset: pulse i_rst_n low across a posedge so the
                    // counter reloads acc = HALF, matching the model's reset().
                    if (rst) begin
                        @(negedge clk);
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    in = a;
                    #1;
                    n = n + 1;
                    if (out !== exp_out) begin
                        $display("FAIL t=%0d in=%b : got %b exp %b", n, a, out, exp_out);
                        fails = fails + 1;
                    end
                    @(posedge clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL relu_cnt: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS relu_cnt: %0d/%0d vectors", n, n);
        else
            $display("FAIL relu_cnt: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
