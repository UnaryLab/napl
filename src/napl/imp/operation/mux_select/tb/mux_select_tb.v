`timescale 1ns/1ps
`default_nettype none
`include "mux_select/vec/mux_select_params.vh"
// Golden rows are <select> <input_a> <input_b> <output>, from the Python model.
// mux_select is combinational, so each row is checked after the inputs settle.
// Co-sim: make test OP=mux_select


module mux_select_tb;
    reg  i_select;
    reg  i_input_a;
    reg  i_input_b;
    wire o_output;

    mux_select dut (
        .i_select (i_select),
        .i_input_a(i_input_a),
        .i_input_b(i_input_b),
        .o_output (o_output)
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
    localparam DATA_COLS  = 4;

    integer fd, code, chars, n, fails;
    reg sel_s, in_a_s, in_b_s, exp_out;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_sel_s;
    reg [8*8-1:0] tok_in_a_s;
    reg [8*8-1:0] tok_in_b_s;
    reg [8*8-1:0] tok_exp_out;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        fd = $fopen("vec/mux_select.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mux_select.vec (run `make vectors` first)");
            $finish;
        end

        n = 0;
        fails = 0;
        // The circuit is a single mux, so the model's zero pp_delay is the only legal value.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL mux_select: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end
        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "mux_select: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s", tok_sel_s, tok_in_a_s, tok_in_b_s, tok_exp_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "mux_select: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "mux_select: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_sel_s !== "0" && tok_sel_s !== "1")
                        $fatal(1, "mux_select: row %0d input field i_select is \"%0s\", expected a single 0 or 1", n + 1, tok_sel_s);
                    sel_s = (tok_sel_s == "1");
                    if (tok_in_a_s !== "0" && tok_in_a_s !== "1")
                        $fatal(1, "mux_select: row %0d input field i_input_a is \"%0s\", expected a single 0 or 1", n + 1, tok_in_a_s);
                    in_a_s = (tok_in_a_s == "1");
                    if (tok_in_b_s !== "0" && tok_in_b_s !== "1")
                        $fatal(1, "mux_select: row %0d input field i_input_b is \"%0s\", expected a single 0 or 1", n + 1, tok_in_b_s);
                    in_b_s = (tok_in_b_s == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "mux_select: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    i_select  = sel_s;
                    i_input_a = in_a_s;
                    i_input_b = in_b_s;
                    #1;
                    n = n + 1;
                    if (o_output !== exp_out) begin
                        $display("FAIL row %0d: i_select=%b i_input_a=%b i_input_b=%b got %b exp %b",
                                 n, i_select, i_input_a, i_input_b, o_output, exp_out);
                        fails = fails + 1;
                    end
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL mux_select: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS mux_select: %0d/%0d vectors", n, n);
        else
            $display("FAIL mux_select: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
