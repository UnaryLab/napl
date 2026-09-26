// Python golden max/arg outputs are checked before each posedge updates state.
// Co-sim: make test OP=max
`timescale 1ns / 1ps
`default_nettype none
`include "max/vec/max_params.vh"


module max_tb;
    reg  clk;
    reg  rst_n;
    reg  in_0;
    reg  in_1;
    wire o_output;
    wire o_index;

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

    integer fd, code, chars, errors, count;
    reg v_reset, v_in_0, v_in_1, v_max, v_arg;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_v_reset;
    reg [8*8-1:0] tok_v_in_0;
    reg [8*8-1:0] tok_v_in_1;
    reg [8*8-1:0] tok_v_max;
    reg [8*8-1:0] tok_v_arg;
    reg [8*LINE_BYTES-1:0] line;

    max dut (
        .i_clk   (clk),
        .i_rst_n (rst_n),
        .i_input_0  (in_0),
        .i_input_1  (in_1),
        .o_output (o_output),
        .o_index  (o_index)
    );

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        errors = 0;
        count  = 0;
        in_0   = 1'b0;
        in_1   = 1'b0;

        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        fd = $fopen("vec/max.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/max.vec");
            $finish;
        end

        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL max: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            $finish;
        end

        // One line holds one row of exactly DATA_COLS binary columns. A line
        // carrying any other column count, a line that fills the line buffer, or
        // an x or z field ends the run with a fatal at that row, so a row can
        // never borrow a column from its neighbour. A blank line is benign only
        // at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "max: row %0d fills the %0d byte line buffer", count + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(
                    line, "%s %s %s %s %s %s",
                    tok_v_reset, tok_v_in_0, tok_v_in_1, tok_v_max, tok_v_arg, extra
                );
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "max: row %0d is blank", count + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "max: row %0d scanned %0d columns, expected %0d", count + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_v_reset !== "0" && tok_v_reset !== "1")
                        $fatal(1, "max: row %0d input field v_reset is \"%0s\", expected a single 0 or 1", count + 1, tok_v_reset);
                    v_reset = (tok_v_reset == "1");
                    if (tok_v_in_0 !== "0" && tok_v_in_0 !== "1")
                        $fatal(1, "max: row %0d input field v_in_0 is \"%0s\", expected a single 0 or 1", count + 1, tok_v_in_0);
                    v_in_0 = (tok_v_in_0 == "1");
                    if (tok_v_in_1 !== "0" && tok_v_in_1 !== "1")
                        $fatal(1, "max: row %0d input field v_in_1 is \"%0s\", expected a single 0 or 1", count + 1, tok_v_in_1);
                    v_in_1 = (tok_v_in_1 == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_v_max !== "0" && tok_v_max !== "1")
                        $fatal(1, "max: row %0d expected field v_max is \"%0s\", expected a single 0 or 1", count + 1, tok_v_max);
                    v_max = (tok_v_max == "1");
                    if (tok_v_arg !== "0" && tok_v_arg !== "1")
                        $fatal(1, "max: row %0d expected field v_arg is \"%0s\", expected a single 0 or 1", count + 1, tok_v_arg);
                    v_arg = (tok_v_arg == "1");
                    @(negedge clk);
                    if (v_reset != 1'b0) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    in_0 = v_in_0;
                    in_1 = v_in_1;
                    #1;
                    if (o_output !== v_max || o_index !== v_arg) begin
                        errors = errors + 1;
                        if (errors <= 10)
                            $display("MISMATCH t=%0d in=(%0d,%0d) got=(%b,%b) exp=(%0d,%0d)",
                                     count, v_in_0, v_in_1, o_output, o_index, v_max, v_arg);
                    end
                    count = count + 1;
                    @(posedge clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (count != `GEN_VECTORS) begin
            $display("FAIL max: compared %0d vectors, generator wrote %0d",
                     count, `GEN_VECTORS);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS max: %0d/%0d vectors", count, `GEN_VECTORS);
        else
            $display("FAIL max: %0d/%0d mismatched", errors, count);
        $finish;
    end
endmodule
`default_nettype wire
