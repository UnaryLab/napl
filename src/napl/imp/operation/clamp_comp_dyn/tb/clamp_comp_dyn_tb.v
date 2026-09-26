`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "clamp_comp_dyn/vec/clamp_comp_dyn_params.vh"
// Golden rows are <rst> <i_input> <i_lo> <i_hi> <o_output>. Reset precedes the
// marked row; the output is checked before the posedge updates state, which is the
// pp_delay 0 the generator carries and the check below pins.
// Co-sim: make test OP=clamp_comp_dyn


module clamp_comp_dyn_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input;
    reg  i_lo;
    reg  i_hi;
    wire o_output;

    clamp_comp_dyn dut (
        .i_clk    (i_clk),
        .i_rst_n  (i_rst_n),
        .i_input  (i_input),
        .i_lo     (i_lo),
        .i_hi     (i_hi),
        .o_output (o_output)
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
    // guessed.
    localparam LINE_BYTES = 32;
    localparam DATA_COLS  = 5;

    integer fd, code, chars, n, fails;
    reg rst_flag, expected;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_flag;
    reg [8*8-1:0] tok_i_input;
    reg [8*8-1:0] tok_i_lo;
    reg [8*8-1:0] tok_i_hi;
    reg [8*8-1:0] tok_expected;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        // The compare below reads the output in the same cycle as the stimulus, so
        // it checks a delay-0 path. A model that grew a register stage would be
        // compared against the wrong cycle and still pass, so the declared delay
        // ends the run here instead.
        if (`GEN_PP_DELAY != 0)
            $fatal(1, "clamp_comp_dyn: the model reports pp_delay %0d, but this testbench compares the output in the stimulus cycle", `GEN_PP_DELAY);

        fd = $fopen("vec/clamp_comp_dyn.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/clamp_comp_dyn.vec (run `make vectors` first)");
            $finish;
        end

        i_input = 1'b0;
        i_lo    = 1'b0;
        i_hi    = 1'b0;
        i_rst_n = 1'b1;

        n = 0;
        fails = 0;
        // One line holds one row of exactly DATA_COLS binary columns. A line
        // carrying any other column count, a line that fills the line buffer, or
        // an x or z data field ends the run with a fatal at that row, so a row
        // can never borrow a column from its neighbour. A blank line is benign
        // only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "clamp_comp_dyn: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s",
                               tok_rst_flag, tok_i_input, tok_i_lo, tok_i_hi, tok_expected, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "clamp_comp_dyn: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "clamp_comp_dyn: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_flag !== "0" && tok_rst_flag !== "1")
                        $fatal(1, "clamp_comp_dyn: row %0d input field rst_flag is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_flag);
                    rst_flag = (tok_rst_flag == "1");
                    if (tok_i_input !== "0" && tok_i_input !== "1")
                        $fatal(1, "clamp_comp_dyn: row %0d input field i_input is \"%0s\", expected a single 0 or 1", n + 1, tok_i_input);
                    i_input = (tok_i_input == "1");
                    if (tok_i_lo !== "0" && tok_i_lo !== "1")
                        $fatal(1, "clamp_comp_dyn: row %0d input field i_lo is \"%0s\", expected a single 0 or 1", n + 1, tok_i_lo);
                    i_lo = (tok_i_lo == "1");
                    if (tok_i_hi !== "0" && tok_i_hi !== "1")
                        $fatal(1, "clamp_comp_dyn: row %0d input field i_hi is \"%0s\", expected a single 0 or 1", n + 1, tok_i_hi);
                    i_hi = (tok_i_hi == "1");
                    // Expected-output column: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_expected !== "0" && tok_expected !== "1")
                        $fatal(1, "clamp_comp_dyn: row %0d expected field is \"%0s\", expected a single 0 or 1", n + 1, tok_expected);
                    expected = (tok_expected == "1");
                    if (rst_flag) begin
                        i_rst_n = 1'b0;
                        @(posedge i_clk);   // async reset fires here (selector counters and latches <= 0)
                        @(negedge i_clk);
                        i_rst_n = 1'b1;
                    end
                    #1;                     // let the combinational outputs settle
                    n = n + 1;
                    if (o_output !== expected) begin
                        $display("FAIL cycle %0d: i_input=%b i_lo=%b i_hi=%b got %b exp %b",
                                 n, i_input, i_lo, i_hi, o_output, expected);
                        fails = fails + 1;
                    end
                    @(posedge i_clk);       // advance the selector state
                    @(negedge i_clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL clamp_comp_dyn: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS clamp_comp_dyn: %0d/%0d vectors (unipolar + bipolar stimulus, static, inverted, and mid-run bands)", n, n);
        else
            $display("FAIL clamp_comp_dyn: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
