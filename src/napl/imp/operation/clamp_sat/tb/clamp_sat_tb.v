`timescale 1ns/1ps
`default_nettype none
// Generated parameters mirror the Python model configuration.
`include "clamp_sat/vec/clamp_sat_params.vh"
// Golden rows are <rst> <unipolar input/output> <bipolar input/output>. Reset
// precedes the marked row; outputs are checked before the posedge updates state.
// Co-sim: make test OP=clamp_sat


module clamp_sat_tb;
    reg  i_clk;
    reg  i_rst_n;
    reg  i_input_uni;
    reg  i_input_bi;
    wire o_uni;
    wire o_bi;

    clamp_sat_unipolar #(
        .LO(`GEN_LO_UNI),
        .HI(`GEN_HI_UNI),
        .FRACWIDTH(`GEN_FRACWIDTH)
    ) dut_uni (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_uni),
        .o_output  (o_uni)
    );

    clamp_sat_bipolar #(
        .LO(`GEN_LO_BI),
        .HI(`GEN_HI_BI),
        .FRACWIDTH(`GEN_FRACWIDTH)
    ) dut_bi (
        .i_clk  (i_clk),
        .i_rst_n(i_rst_n),
        .i_input   (i_input_bi),
        .o_output  (o_bi)
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
    reg rst_flag, exp_uni, exp_bi;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst_flag;
    reg [8*8-1:0] tok_i_input_uni;
    reg [8*8-1:0] tok_exp_uni;
    reg [8*8-1:0] tok_i_input_bi;
    reg [8*8-1:0] tok_exp_bi;
    reg [8*LINE_BYTES-1:0] line;

    initial begin
        fd = $fopen("vec/clamp_sat.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/clamp_sat.vec (run `make vectors` first)");
            $finish;
        end

        i_input_uni = 1'b0;
        i_input_bi  = 1'b0;
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
                    $fatal(1, "clamp_sat: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s %s %s",
                               tok_rst_flag, tok_i_input_uni, tok_exp_uni, tok_i_input_bi, tok_exp_bi, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "clamp_sat: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "clamp_sat: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst_flag !== "0" && tok_rst_flag !== "1")
                        $fatal(1, "clamp_sat: row %0d input field rst_flag is \"%0s\", expected a single 0 or 1", n + 1, tok_rst_flag);
                    rst_flag = (tok_rst_flag == "1");
                    if (tok_i_input_uni !== "0" && tok_i_input_uni !== "1")
                        $fatal(1, "clamp_sat: row %0d input field i_input_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_i_input_uni);
                    i_input_uni = (tok_i_input_uni == "1");
                    if (tok_i_input_bi !== "0" && tok_i_input_bi !== "1")
                        $fatal(1, "clamp_sat: row %0d input field i_input_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_i_input_bi);
                    i_input_bi = (tok_i_input_bi == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_uni !== "0" && tok_exp_uni !== "1")
                        $fatal(1, "clamp_sat: row %0d expected field exp_uni is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_uni);
                    exp_uni = (tok_exp_uni == "1");
                    if (tok_exp_bi !== "0" && tok_exp_bi !== "1")
                        $fatal(1, "clamp_sat: row %0d expected field exp_bi is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_bi);
                    exp_bi = (tok_exp_bi == "1");
                    if (rst_flag) begin
                        i_rst_n = 1'b0;
                        @(posedge i_clk);   // async reset fires here (stage accumulators and sequences <= 0)
                        @(negedge i_clk);
                        i_rst_n = 1'b1;
                    end
                    #1;                     // let the combinational outputs settle
                    n = n + 1;
                    if (o_uni !== exp_uni) begin
                        $display("FAIL cycle %0d unipolar: i_input=%b got %b exp %b", n, i_input_uni, o_uni, exp_uni);
                        fails = fails + 1;
                    end
                    if (o_bi !== exp_bi) begin
                        $display("FAIL cycle %0d bipolar: i_input=%b got %b exp %b", n, i_input_bi, o_bi, exp_bi);
                        fails = fails + 1;
                    end
                    @(posedge i_clk);       // advance the stage and sequence state
                    @(negedge i_clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL clamp_sat: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS clamp_sat: %0d/%0d vectors (unipolar + bipolar)", n, n);
        else
            $display("FAIL clamp_sat: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
