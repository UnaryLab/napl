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
    localparam LINE_BYTES = 16;
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg [8*8-1:0] tok, extra;
    reg [8*8-1:0] tok_k;
    reg [8*8-1:0] tok_exp_q;
    reg [8*LINE_BYTES-1:0] line;
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
        // A row carrying any other column count, a row that fills the line
        // buffer, a data row whose tag is not 0 or 1, or an x or z data field ends
        // the run with a fatal at that row, so a row can never borrow a column
        // from its neighbour. A blank line is benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "jkff: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok, tok_k, tok_exp_q, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "jkff: row %0d is blank", n + 1);
                end else if (code == 1 && tok == "R") begin
                    i_rst_n   = 1'b0;
                    i_input_j = 1'b0;
                    i_input_k = 1'b0;
                    @(posedge i_clk);
                    @(negedge i_clk);
                    i_rst_n = 1'b1;
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "jkff: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    if (tok !== "0" && tok !== "1")
                        $fatal(1, "jkff: row %0d tag column is \"%0s\", expected 0 or 1 on a data row, with R alone on its own line", n + 1, tok);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_k !== "0" && tok_k !== "1")
                        $fatal(1, "jkff: row %0d input field k is \"%0s\", expected a single 0 or 1", n + 1, tok_k);
                    k = (tok_k == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_q !== "0" && tok_exp_q !== "1")
                        $fatal(1, "jkff: row %0d expected field exp_q is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_q);
                    exp_q = (tok_exp_q == "1");
                    j = (tok[7:0] == "1");
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
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL jkff: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS jkff: %0d/%0d vectors", n, n);
        else
            $display("FAIL jkff: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
