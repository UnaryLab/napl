`timescale 1ns/1ps
`default_nettype none
// Generated DEPTH mirrors the Python model configuration.
`include "delay/vec/delay_params.vh"
// The Python golden output is the oldest cell before the posedge shifts input.
// Two DUTs run side by side, one per init mode, so both hardware forms INIT
// selects are compared against the model on the same drive.
// R requests an active-low reset that reloads each DUT's initial pattern.
// Co-sim: make test OP=delay


module delay_tb;
    reg  clk;
    reg  rst_n;
    reg  in_bit;
    wire out_zero;
    wire out_alt;

    delay #(.DEPTH(`GEN_DEPTH), .INIT(0)) dut_zero (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input  (in_bit),
        .o_output (out_zero)
    );

    delay #(.DEPTH(`GEN_DEPTH), .INIT(1)) dut_alt (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input  (in_bit),
        .o_output (out_alt)
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
    localparam LINE_BYTES = 32;
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg [8*8-1:0] tok, extra;
    reg [8*8-1:0] tok_exp_zero, tok_exp_alt;
    reg a, exp_zero, exp_alt;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        in_bit = 1'b0;

        // Assert active-low reset across a clock edge to load each DUT's initial
        // pattern (the async reset fires on the posedge while rst_n is low), then
        // release it on a negedge. No posedge may occur between release and the
        // first check, or the delay line would shift once.
        rst_n = 1'b0;
        @(negedge clk);
        @(posedge clk);   // async reset loads the initial pattern here
        @(negedge clk);
        rst_n = 1'b1;     // released; next posedge (inside the loop) is timestep 1

        fd = $fopen("vec/delay.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/delay.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        if (`GEN_PP_DELAY != `GEN_DEPTH) begin
            $display(
                "FAIL delay: generator macros disagree, GEN_DEPTH %0d, GEN_PP_DELAY %0d",
                `GEN_DEPTH,
                `GEN_PP_DELAY
            );
            fails = fails + 1;
        end
        // One line holds one row: either the reset sentinel "R" alone, or a tag
        // column of 0 or 1 followed by exactly DATA_COLS-1 binary data columns. A
        // line carrying any other column count, a line that fills the line buffer,
        // a data row whose tag is not 0 or 1, or an x or z data field ends the run
        // with a fatal at that row, so a row can never borrow a column from its
        // neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            // Rows are "<in> <out_zero> <out_alt>" or the reset sentinel "R".
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "delay: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok, tok_exp_zero, tok_exp_alt, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "delay: row %0d is blank", n + 1);
                end else if (code == 1 && tok == "R") begin
                    // Mid-stream reset: pulse i_rst_n low across a posedge to reload
                    // each initial pattern, mirroring model.reset().
                    rst_n  = 1'b0;
                    in_bit = 1'b0;
                    @(posedge clk);   // async reset reloads the initial pattern
                    @(negedge clk);
                    rst_n  = 1'b1;    // released; back on a negedge, registers stable
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "delay: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (tok !== "0" && tok !== "1")
                        $fatal(1, "delay: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with R alone on its own line", n + 1, tok);
                    a = (tok == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the columns as characters instead.
                    if (tok_exp_zero !== "0" && tok_exp_zero !== "1")
                        $fatal(1, "delay: row %0d expected field exp_zero is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_zero);
                    if (tok_exp_alt !== "0" && tok_exp_alt !== "1")
                        $fatal(1, "delay: row %0d expected field exp_alt is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_alt);
                    exp_zero = (tok_exp_zero == "1");
                    exp_alt  = (tok_exp_alt == "1");
                    n = n + 1;
                    if (out_zero !== exp_zero) begin
                        $display("FAIL cyc=%0d init=zero in=%b : got %b exp %b", n, a, out_zero, exp_zero);
                        fails = fails + 1;
                    end
                    if (out_alt !== exp_alt) begin
                        $display("FAIL cyc=%0d init=alternate in=%b : got %b exp %b", n, a, out_alt, exp_alt);
                        fails = fails + 1;
                    end
                    in_bit = a;
                    @(posedge clk);   // shift: emit oldest, append in_bit
                    @(negedge clk);   // settle for the next check
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL delay: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS delay: %0d/%0d vectors", n, n);
        else
            $display("FAIL delay: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
