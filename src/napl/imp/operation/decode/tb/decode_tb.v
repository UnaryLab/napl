`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "decode/vec/decode_params.vh"
// Python golden count is checked in the arrival cycle, before the posedge stores
// it. The reset column requests active-low reset to a zero count before its row.
// Co-sim: make test OP=decode


module decode_tb;
    localparam WIDTH = `GEN_WIDTH;

    reg  clk, rst_n, spike;
    wire [WIDTH:0] spike_count;

    decode #(.WIDTH(WIDTH)) dut (
        .i_clk         (clk),
        .i_rst_n       (rst_n),
        .i_input       (spike),
        .o_spike_count (spike_count)
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

    // Every column is consumed through a sink of a fixed width, so a token wider
    // than its sink is rejected below rather than silently truncated at the
    // sink. COUNT_MAX is the widest value o_spike_count can carry; the two
    // stimulus columns are the same rule at a one-bit sink.
    localparam COUNT_MAX = (1 << (WIDTH + 1)) - 1;

    integer fd, code, chars, n, fails;
    reg     a, rst;
    integer exp_count;
    // The one-bit columns are scanned as text so a token outside {0, 1} always
    // reaches the guard below; the reg is wider than a legal token so an
    // over-wide token reads back over-wide instead of saturating.
    reg [8*8-1:0] tok_a;
    reg [8*8-1:0] tok_rst;
    reg [8*8-1:0] extra;
    reg [8*LINE_BYTES-1:0] line;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        spike = 1'b0;
        rst_n = 1'b1;
        n     = 0;
        fails = 0;

        fd = $fopen("vec/decode.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/decode.vec (run `make vectors` first)");
            $finish;
        end

        // pp_delay is 0: the count is combinational in the arrival cycle.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL decode: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // Reset to the model's post-reset() state (count = 0). Hold rst_n low
        // across a posedge, then deassert on the falling edge so the first
        // driven vector sees count = 0 (no stray advancing edge in between).
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // One line holds one row of exactly DATA_COLS columns: two binary columns
        // around a decimal count. A line carrying any other column count, a line
        // that fills the line buffer, or an x or z data field ends the run with a
        // fatal at that row, so a row can never borrow a column from its
        // neighbour. The count column is tested with a reduction xor, which is x
        // when any bit of the scanned value is x or z. A blank line is benign
        // only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "decode: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count in this format string must match DATA_COLS.
                code = $sscanf(line, "%s %d %s %s", tok_a, exp_count, tok_rst, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "decode: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "decode: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // INPUT COLUMN GUARDS: a stimulus column outside {0, 1} is a
                    // corrupt file, and is named as one rather than truncated
                    // into a bit the DUT would accept as stimulus.
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "decode: row %0d input field spike is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "decode: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    // EXPECTED COLUMN GUARD: the count is compared through a
                    // WIDTH+1 bit sink, so a golden count wider than that sink is
                    // named as a corrupt answer rather than truncated into a
                    // count that can match the DUT.
                    if ((^exp_count) === 1'bx)
                        $fatal(1, "decode: row %0d field exp_count is unknown (%b)", n + 1, exp_count);
                    if (exp_count < 0 || exp_count > COUNT_MAX)
                        $fatal(1, "decode: row %0d field exp_count is out of range (%0d), the %0d bit count holds 0 to %0d",
                               n + 1, exp_count, WIDTH + 1, COUNT_MAX);
                    if (rst != 0) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    // Drive from the negedge: the count register samples i_input on
                    // the posedge, so an input changing at that edge would race it.
                    spike = a;
                    #1;
                    n = n + 1;
                    if (spike_count !== exp_count[WIDTH:0]) begin
                        $display("FAIL t=%0d spike=%b : got %0d exp %0d", n, a, spike_count, exp_count);
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
            $display("FAIL decode: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS decode: %0d/%0d vectors", n, n);
        else
            $display("FAIL decode: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire
