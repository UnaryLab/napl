`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH/TRIGGER mirror the Python model configuration.
`include "encode_hold/vec/encode_hold_params.vh"
// Python golden rows are <rst> <i_input> <o_output>; the output is checked in the
// arrival cycle, before the posedge advances state. rst=1 requests an active-low
// reset to the model's post-reset() state before its row.
// Co-sim: make test OP=encode_hold


module encode_hold_tb;
    localparam WIDTH   = `GEN_WIDTH;
    localparam TRIGGER = `GEN_TRIGGER;

    reg  clk, rst_n, spike;
    wire out;

    encode_hold #(.WIDTH(WIDTH), .TRIGGER(TRIGGER)) dut (
        .i_clk    (clk),
        .i_rst_n  (rst_n),
        .i_input  (spike),
        .o_output (out)
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
    localparam DATA_COLS  = 3;

    integer fd, code, chars, n, fails;
    reg a, rst, exp_out;
    reg [8*8-1:0] extra;
    reg [8*8-1:0] tok_rst;
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_exp_out;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        spike = 1'b0;
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/encode_hold.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/encode_hold.vec (run `make vectors` first)");
            $finish;
        end

        // pp_delay is 0: the output is combinational in the arrival cycle.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL encode_hold: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // Reset to the model's post-reset() state. Hold rst_n low across a posedge,
        // then deassert on the falling edge so the first driven vector starts at t=0
        // with no stray advancing edge in between.
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

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
                    $fatal(1, "encode_hold: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %s %s", tok_rst, tok_a, tok_exp_out, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "encode_hold: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "encode_hold: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // Input columns: a token wider than one bit is a corrupt vec file,
                    // so it is rejected here rather than truncated into stimulus.
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "encode_hold: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "encode_hold: row %0d input field a is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    // Expected-output columns: a token wider than one bit would be
                    // truncated into a bit that can match the DUT, so a wrong answer
                    // would be accepted. Compare the column as characters instead.
                    if (tok_exp_out !== "0" && tok_exp_out !== "1")
                        $fatal(1, "encode_hold: row %0d expected field exp_out is \"%0s\", expected a single 0 or 1", n + 1, tok_exp_out);
                    exp_out = (tok_exp_out == "1");
                    if (rst) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    // Drive from the negedge: the state registers sample on the posedge,
                    // so an input changing at that edge would race them.
                    spike = a;
                    #1;
                    n = n + 1;
                    if (out !== exp_out) begin
                        $display("FAIL t=%0d spike=%b : got %b exp %b", n, a, out, exp_out);
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
            $display("FAIL encode_hold: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS encode_hold: %0d/%0d vectors (unipolar+bipolar)", n, n);
        else
            $display("FAIL encode_hold: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
