`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH mirrors the Python model configuration.
`include "max_sync/vec/max_sync_params.vh"
// Python golden outputs use the pre-update counter and are checked before each
// posedge. RST requests active-low reset before replay continues.
// Co-sim: make test OP=max_sync


module max_sync_tb;
    reg  clk;
    reg  rst_n;
    reg  in_0, in_1;
    wire out;

    max_sync #(.DEPTH(`GEN_DEPTH)) dut (
        .i_clk     (clk),
        .i_rst_n   (rst_n),
        .i_input_0 (in_0),
        .i_input_1 (in_1),
        .o_output  (out)
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
    // guessed. Reset rows are the rows that do not carry DATA_COLS columns:
    // each holds its tag alone and is accepted ahead of the column check.
    localparam LINE_BYTES = 128;
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg a, b, exp_out;
    reg [8*8-1:0] tag, extra;
    reg [8*8-1:0] tok_b;
    reg [8*8-1:0] tok_exp_out;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_0 = 1'b0;
        in_1 = 1'b0;

        // Release reset on a negedge so no update precedes the first check.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/max_sync.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/max_sync.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // The output is combinational in the pre-update counter.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL max_sync: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        // One line holds one row: either the reset sentinel "RST" alone, or a tag
        // column of 0 or 1 followed by exactly 2 binary data columns. A line
        // carrying any other column count, a line that fills the line buffer, a
        // data row whose tag is not 0 or 1, or an x or z data field ends the run
        // with a fatal at that row, so a row can never borrow a column from its
        // neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            // Rows are "in_0 in_1 exp_out" or the reset sentinel "RST".
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "max_sync: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                code = $sscanf(line, "%s %s %s %s", tag, tok_b, tok_exp_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "max_sync: row %0d is blank", n + 1);
                end else if (code == 1 && tag == "RST") begin
                    rst_n = 1'b0;
                    @(posedge clk);
                    @(negedge clk);
                    rst_n = 1'b1;
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "max_sync: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (tag !== "0" && tag !== "1")
                        $fatal(1, "max_sync: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with RST alone on its own line", n + 1, tag);
                    a = (tag == "1");
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_b !== "0" && tok_b !== "1")
                        $fatal(1, "max_sync: row %0d input field in_1 is \"%0s\", expected a single 0 or 1", n + 1, tok_b);
                    b = (tok_b == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "max_sync: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    in_0 = a;
                    in_1 = b;
                    #1;
                    n = n + 1;
                    if (out !== exp_out) begin
                        $display("FAIL cyc=%0d in_0=%b in_1=%b : out got %b exp %b", n, a, b, out, exp_out);
                        fails = fails + 1;
                    end
                    @(posedge clk);
                    @(negedge clk);
                end
            end
        end
        $fclose(fd);

        // A vec file that lost or gained rows fails instead of passing on the rest.
        if (n != `GEN_VECTORS) begin
            $display("FAIL max_sync: consumed %0d rows, generator emitted %0d", n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS max_sync: %0d/%0d vectors", n, n);
        else
            $display("FAIL max_sync: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
